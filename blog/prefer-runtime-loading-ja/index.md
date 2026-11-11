---
title: MPI・HDF5・BLAS・CUDA はビルド時にリンクしない
---

# MPI・HDF5・BLAS・CUDA はビルド時にリンクしない

*tensor4all のネイティブライブラリは，MPI，HDF5，CUDA，BLAS をビルド時にリンクせず，実行時にロードします．本記事では，その理由と，我々が使っている二つの方式（`dlopen` と関数ポインタの注入），そしてどちらをいつ使うかを説明します．*

by **品岡 寛**（埼玉大学），tensor4all チームを代表して

*🌐 [English](https://tensor4all.org/blog/prefer-runtime-loading/) · [日本語](https://tensor4all.org/blog/prefer-runtime-loading-ja/)*

---

## ビルド時にリンクすると何が困るか

C や Fortran のライブラリをコンパイル言語から使うときは，ふつうビルド時にリンクします．ビルドした計算機でそのまま動かすなら困りません．困るのは，バイナリを配布したいときと，ライブラリを Julia や Python から呼ぶときです．

ある MPI 実装，ある HDF5，ある CUDA にリンクしたバイナリは，同じライブラリが入った環境でしか動かず，配布すべき組み合わせはすぐに増えます．ビルドにもツールチェーンとヘッダが要ります．Rust から bindgen で MPI をリンクするには C コンパイラ，MPI のヘッダ，libclang が必要で，パッケージを入れたいだけの人には負担が大きすぎます．

Julia や Python の中では，別の問題も起きます．Julia は常に BLAS をロードしていて，HDF5.jl を使えば HDF5 もロードします．Python でも NumPy，h5py，mpi4py がすでに読み込まれていることがあります．そこへ我々のライブラリが自前のコピーを持ち込むと，一つのプロセスに HDF5 の状態が二つ，あるいは BLAS の実装とスレッドプールが二つできます．さらに，どの MPI が動いているか，どの BLAS が有効か，その整数が32ビットか64ビットかは，ホスト言語が知っています．ビルド時にリンクしたライブラリには，それを尋ねる手段がありません．

そこで，使うライブラリは実行時に決め，プログラムの起動時に教えてもらうことにしています．バイナリは配布しやすくなり，Julia や Python のラッパーも書きやすくなります．ホストが，すでに持っている MPI ランタイム，BLAS，HDF5 をそのまま渡せるからです．

方式は二つあり，向いているライブラリの種類が違います．

## 方式1：`dlopen` でライブラリを開く

ライブラリが実行時に共有オブジェクトを開き，必要なシンボルを探します．HDF5，MPI，CUDA ではこの方式です．

[hdf5-rt](https://github.com/tensor4all/hdf5-rt) は hdf5-metno のフォークで，HDF5 を `dlopen` でロードします．`init` はパスを任意で受け取るので，ホストは自分が使っている HDF5 を指定できます（Julia なら `HDF5_jll.libhdf5_path` など）．パスがなければ，プラットフォームごとのよくある場所を順に試します．ロード後には HDF5 のバージョンを確認し，古すぎるものは拒否します．ライブラリのハンドルはわざとリークさせ，`dlclose` しません．内部状態が残ったままアンロードされると，HDF5 の後始末が無限ループに入ったりクラッシュしたりすることがあるためです．

[rsmpi-rt](https://github.com/tensor4all/rsmpi-rt) は rsmpi のフォークで，[MPItrampoline](https://github.com/eschnett/MPItrampoline) の MPIABI に基づく `mpi-rt-sys-backend` を追加しています．MPIABI は，MPI 実装ごとに異なる ABI の部分を固定します．ハンドル型は整数（Rust 側では `usize`），`MPI_Status` のレイアウトは固定，定数はマクロではなくシンボルとして公開されます．実行時には `MPI_RT_LIB` で指定した [MPIwrapper](https://github.com/eschnett/MPIwrapper) のライブラリをロードし，MPIwrapper がネイティブの MPI へ転送します．これで Rust のビルドに C コンパイラも MPI ヘッダも libclang も要らなくなります．その代わり，MPIwrapper をシステムの MPI に対して一度ビルドする必要があります．狙いは，Rust の MPI コードを Julia や Python から呼び，[MPI.jl](https://github.com/JuliaParallel/MPI.jl) や [mpi4py](https://mpi4py.readthedocs.io/) と一緒に動かすことです．現在の相互運用テストでは，Rust，Python，Julia のプロセスを一つの `mpiexec` で起動し，`MPI_COMM_WORLD` を共有させています．まだ実験段階で，いずれ rsmpi 本体に還元したいと考えています．

CUDA については，tenferro-rs が `cudarc` の `dynamic-loading` feature を使い，CUDA のドライバ，ランタイム，NVRTC を実行時に探します．GPU カーネルは [t4a-cuda-kernels](https://github.com/tensor4all/t4a-cuda-kernels) からソースとして取り込み，実行時に NVRTC でコンパイルするので，バイナリは CUDA にリンクしていません．

`dlopen` が向くのは，ライブラリが一つで，その場所をホストが教えられるか決まった場所にあり，シンボルが名前で束縛できる程度に安定している場合です．

## 方式2：ホストが関数ポインタを渡す

Julia や Python から使う BLAS と LAPACK については，我々の知る限り，安定した方法は注入しかありません．ホスト言語が，自分の使っている BLAS の関数を調べてポインタを Rust のライブラリに渡し，ライブラリはそれを経由して呼び出します．

こちらから `dlopen("libblas")` してもうまくいかない理由は三つあります．一つ目に，探す場所が決まっていません．NumPy と SciPy の wheel は OpenBLAS を wheel の中に同梱しています．Julia は [libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline) を通して BLAS を呼び，各呼び出しはロード済みのバックエンド（既定は OpenBLAS，ユーザが切り替えれば MKL や Accelerate）へ転送されます．二つ目に，システムの BLAS があったとしても，ホストの BLAS の隣にロードすれば，一つのプロセスに実装とスレッドプールが二つ載ります．三つ目に，LP64（32ビット整数）と ILP64（64ビット整数）の BLAS は ABI が違い，しかもシンボル名からは確実に区別できません．ILP64 のビルドには `64_` を付けるものも，付けないものもあります．どちらなのかを知っているのはホストです．

### sparse-ir-rs

最も分かりやすい例は，SpM-lab で開発している，虚時間の伝搬関数の中間表現（IR）の Rust 実装 [sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs) です．Julia，Python，C++，Fortran からは C API を通して使われています．

中核の行列積は，BLAS の流儀で書いた列優先の GEMM（`C ← α·op(A)·op(B) + β·C`，leading dimension は明示）を通ります．既定のバックエンドは純 Rust の faer です．BLAS は実行時に注入でき，Rust からは `set_blas_backend`（LP64）か `set_ilp64_backend`（ILP64）を呼び，`clear_blas_backend` で faer に戻します．C API には `spir_gemm_backend_new_from_fblas_lp64` と `_ilp64` があり，`dgemm` と `zgemm` のポインタを受け取ります．グローバルな既定値を設定する代わりに，呼び出しごとにバックエンドのハンドルを渡すこともできます．

二つのラッパーは，これを次のように使っています．

- Python：SciPy が入っていれば，`pylibsparseir` が `scipy.linalg.cython_blas` の公開する PyCapsule から `dgemm` と `zgemm` のポインタを取り出し，LP64 のバックエンドとして登録します．
- Julia：SparseIR.jl が `BLAS.USE_BLAS64` を見て ILP64 か LP64 のインタフェースを選び，`BLAS.lbt_get_forward` で libblastrampoline から転送先の関数を取得して，対応するバックエンドを作ります．

これで同じ Rust のコードが，SciPy や Julia の使っている BLAS を正しい整数幅で呼び，プロセス内の BLAS は一つで済みます．

### tenferro-rs，cblas-inject，lapack-inject

テンソルネットワークのコードには，SVD，QR，固有値問題のための LAPACK も必要です．[lapack-inject](https://github.com/shinaoka/lapack-inject) は，`lapack-sys` が宣言する Fortran の LAPACK シンボル 1315 個をすべてエクスポートするので，`lapack-src` のシンボルを期待するクレートはこれにリンクできます．各シンボルは実行時に登録されたポインタへ転送し，ルーチンごとに LP64 用と ILP64 用の登録関数（`register_dgesv_lp64`，`register_dgesv_ilp64` など）があります．BLAS について同じことをするのが [cblas-inject](https://crates.io/crates/cblas-inject) です．

tenferro-rs は，`tenferro-cpu` の `provider-inject` feature の下でこの二つを使っています．tenferro 自身は32ビット整数で BLAS と LAPACK を呼びます．ホストは `register_blas_gemm_provider_ptrs` と `register_lapack_provider_ptrs` で，LP64 か ILP64 かを指定してポインタを登録し，ILP64 の場合は inject 側のクレートが整数引数を変換します．64ビットの libblastrampoline を使う Julia からでも，BLAS をそのまま tenferro に渡せます．

## どちらを使うか

| | `dlopen` | ポインタの注入 |
|---|---|---|
| ライブラリを見つけるのは | 我々のライブラリ | ホスト言語 |
| 向いているもの | 場所が分かり，シンボルが安定しているライブラリ（HDF5，MPIwrapper 経由の MPI，CUDA） | ホストがすでに持っていて，差し替えうるライブラリ（Julia と Python の BLAS，LAPACK） |
| 例 | hdf5-rt，rsmpi-rt，cudarc | sparse-ir-rs，cblas-inject，lapack-inject |
| ホストが渡すもの | パス（既定で見つかるなら不要） | 関数ポインタとその整数幅 |

Julia や Python から使う BLAS と LAPACK について，注入に代わる安定した方法を我々は知りません．それ以外は，ホストが設定できるパスを使った `dlopen` を既定にするのがよいと考えています．どちらの方式でも，ライブラリはフックを公開し，何もインストールしなくても動くよう純 Rust のフォールバック（sparse-ir-rs では faer）を用意し，より速いライブラリの選択は利用者に任せます．

## 実際にハマったところ

- 整数幅．LP64 を期待するところへ ILP64 のポインタを渡すと（逆も同じ），エラーにならず，結果がおかしくなるかクラッシュします．Julia では `BLAS.USE_BLAS64` を確認します．SciPy の `cython_blas` のポインタは LP64 です．
- スレッド安全性と寿命．Rust 側は注入された関数を複数のスレッドから呼び，バックエンドが生きている間ポインタを保持します．提供側はスレッドセーフで，ロードされたままでなければなりません．
- Fortran の呼び出し規約．引数はすべて参照渡しで，`'N'` や `'T'` のような文字引数には，Fortran コンパイラが隠れた長さ引数を付けます．こうした細部は，ラッパーごとに書かず，テストされた一つのクレートにまとめるべきです．
- アンロード．HDF5 のように大域的な状態を持つライブラリは `dlclose` に耐えないことがあるので，ハンドルはリークさせます．
- テストしていない組み合わせ．sparse-ir-rs は C API のテストを，既定のバックエンド，OpenBLAS LP64，OpenBLAS ILP64 で走らせています．rsmpi-rt は mpi4py と MPI.jl との相互運用テストを走らせています．これ以外の組み合わせは，テストするまでは壊れているものとして扱っています．

## エージェントが書くコード

我々の Rust コードの大半はコーディングエージェントが書き，それを機械で検査しています．詳しくは[誰も逐行で読まないコードをどう検証しているか](https://tensor4all.org/blog/how-we-check-agent-written-code-ja/)に書きました．「BLAS，LAPACK，MPI，HDF5，CUDA をビルド時にリンクしない」は機械で検査できる規則です．たとえば，ビルドスクリプトがこれらのライブラリへのリンク指示を出したら CI を失敗させればよいのです．

## リンク

- [hdf5-rt](https://github.com/tensor4all/hdf5-rt) · [rsmpi-rt](https://github.com/tensor4all/rsmpi-rt) · [lapack-inject](https://github.com/shinaoka/lapack-inject) · [cblas-inject](https://crates.io/crates/cblas-inject)
- [tenferro-rs](https://github.com/tensor4all/tenferro-rs) · [sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs)

Julia や Python の利用者に届けにくいネイティブライブラリをお持ちでしたら，[Matrix](https://tensor4all.org/matrix.html) で教えてください．
