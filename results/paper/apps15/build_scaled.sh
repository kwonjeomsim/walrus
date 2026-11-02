#!/bin/bash
# Builds the six held-out apps as single runs (REPS=1) with larger inputs.
# Sizes come from the environment, defaults are the original driver values.
set -eu
XZ_N=${XZ_N:-1048576}; BERRY_S=${BERRY_S:-1}; JANET_S=${JANET_S:-1}
TCC_NFUNC=${TCC_NFUNC:-400}; LA_NFILES=${LA_NFILES:-4000}; PNG_W=${PNG_W:-512}; PNG_H=${PNG_H:-$PNG_W}
SDK=/opt/wasi-sdk
CC="$SDK/bin/clang --sysroot $SDK/share/wasi-sysroot"
A=$HOME/projects/wasmDataset/external/apps
D=$A/drivers; BD=$A/build
OUT=${OUT_DIR:?}; S=$OUT/src; mkdir -p $S
EMU="-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_GETPID"
L="-lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-mman -lwasi-emulated-getpid"
F="-O2 -w $EMU"; STACK="-Wl,-z,stack-size=8388608"
only=" ${*:-xz berry janet tinycc libarchive libpng} "

if [[ $only == *" xz "* ]]; then
sed "s/^#define N (1 << 20)/#define N ((size_t)$XZ_N)/" $D/d_xz.c > $S/d_xz.c
$CC $F -DREPS=1 -I$A/xz/src/liblzma/api -o $OUT/app_xz.wasm $S/d_xz.c $BD/xz/liblzma.a
fi

if [[ $only == *" berry "* ]]; then
BB=$BD/berry
sed -e "s/0\\.\\.39999 l/0..$((40000*BERRY_S-1)) l/" -e "s/0\\.\\.29999 m/0..$((30000*BERRY_S-1)) m/" \
    -e "s/0\\.\\.1999 st/0..$((2000*BERRY_S-1)) st/" -e "s/0\\.\\.19999 acc/0..$((20000*BERRY_S-1)) acc/" $D/d_berry.c > $S/d_berry.c
$CC $F -std=c99 -DREPS=1 -mllvm -wasm-enable-sjlj -mllvm -wasm-use-legacy-eh=false --no-wasm-opt \
    -I$BB/conf -I$BB/inc -I$A/berry/src -I$A/berry/default $STACK -o $OUT/app_berry.wasm \
    $A/berry/src/*.c $A/berry/default/be_modtab.c $A/berry/default/be_port.c $S/d_berry.c $L -lsetjmp -lm
fi

if [[ $only == *" janet "* ]]; then
JB=$BD/janet
JD="-DJANET_REDUCED_OS -DJANET_SINGLE_THREADED -DJANET_NO_DYNAMIC_MODULES -DJANET_NO_THREADS -DJANET_NO_EV -DJANET_NO_NET \
    -DJANET_NO_PROCESSES -DJANET_NO_FFI -DJANET_NO_REALPATH -DJANET_NO_SYMLINKS -DJANET_NO_UMASK -DJANET_NO_INTERPRETER_INTERRUPT"
sed -e "s/:range \[0 40000\]/:range [0 $((40000*JANET_S))]/" -e "s/(for i 0 30000/(for i 0 $((30000*JANET_S))/g" \
    -e "s/(for i 0 3000 /(for i 0 $((3000*JANET_S)) /" $D/d_janet.c > $S/d_janet.c
$CC $F -Wno-error=implicit-function-declaration $JD -DREPS=1 -I$JB/shim -I$D/shim -I$JB/src/src/include \
    -I$JB/src/src/conf $STACK -o $OUT/app_janet.wasm $JB/src/build/c/janet.c $S/d_janet.c $L -lm
fi

if [[ $only == *" tinycc "* ]]; then
sed -e "s/^#define NFUNC 400/#define NFUNC $TCC_NFUNC/" -e "s/size_t cap = 1 << 22/size_t cap = (size_t)$TCC_NFUNC * 1024 + 4096/" \
    $D/d_tinycc.c > $S/d_tinycc.c
$CC -O2 -w -DREPS=1 -DTCC_TARGET_X86_64 -DONE_SOURCE=1 -DCONFIG_TCC_STATIC -I$D/shim -I$D/tinycc_gen -I$A/tinycc \
    $EMU $STACK -o $OUT/app_tinycc.wasm $S/d_tinycc.c $L -lm
fi

if [[ $only == *" libarchive "* ]]; then
sed -e "s/^#define NFILES 4000/#define NFILES $LA_NFILES/" -e "s/^#define CAP (64u << 20)/#define CAP (512u << 20)/" \
    $D/d_libarchive.c > $S/d_libarchive.c
$CC -O2 -w -DREPS=1 -I$A/libarchive/libarchive $EMU -o $OUT/app_libarchive.wasm $S/d_libarchive.c $BD/libarchive/obj/*.o $L
fi

if [[ $only == *" libpng "* ]]; then
sed -e "s/const int W=512,H=512;/const int W=$PNG_W,H=$PNG_H;/" -e "s/for(int r=0;r<15;r++)/for(int r=0;r<1;r++)/" \
    $D/d_libpng.c > $S/d_libpng.c
ZS=$(ls $A/zlib/*.c | grep -v -E "gz|example|minigzip"); PS=$(ls $A/libpng/png*.c | grep -v -E "pngtest|example")
$CC -O2 -w -I$D/shim -I$A/libpng -I$A/zlib -o $OUT/app_libpng.wasm $S/d_libpng.c $PS $ZS
fi
