---
title: tenferro-rs のご紹介
---

# Julia から Rust へ：AI エージェント時代の科学計算向け微分可能テンソルスタック

*[tenferro-rs](https://github.com/tensor4all/tenferro-rs) は Rust ネイティブな密テンソルスタックだ。線形代数、PyTorch 流の eager 自動微分、JAX 流の traced 変換、NumPy 流の einsum、FFT、拡張可能な演算 crate 群と、明示的な CPU/CUDA バックエンドを備える。最初の crate 群は 2026 年 6 月 23 日（JST）に crates.io へ公開した。*

文責: **Hiroshi Shinaoka**（埼玉大学）、tensor4all チーム

*🌐 [English](https://tensor4all.org/blog/introducing-tenferro-rs/) · [日本語](https://tensor4all.org/blog/introducing-tenferro-rs-ja/) · [简体中文](https://tensor4all.org/blog/introducing-tenferro-rs-zh/)*

![tenferro-rs architecture overview](tenferro-architecture.svg)

---

テンソルネットワークのコードは、これまで Julia で書かれることが多かった。私たちも例外ではない。ITensors とその周辺エコシステムでは、数式に近い形でプロトタイプを書ける。試行錯誤もしやすい。IR 基底、スパースモデリング、そして [tensor4all](https://tensor4all.org) のテンソルクロス補間（TCI）／quantics スタックも、まず Julia で開発してきた。

一方で、コードベースが大きくなると Julia での開発はだんだん重くなる。実行時になって初めて出る型不安定性、編集とテストの往復を遅くするコンパイル／プリコンパイル時間、そしてコードが増えるほど正しさを確認しにくくなる感覚がある。テンソルネットワークのスタックをより大きなシステムへ組み込む段階で、この問題を無視できなくなった。そこで、計算エンジンを Rust へ移し始めた。

始めてすぐ、別の問題にも気づいた。土台にしたいテンソルライブラリが、まだ揃っていなかった。Rust にも用途ごとのライブラリはある。配列なら ndarray、深層学習なら Burn、線形代数なら faer。ただ、自動微分から einsum までをまとめて扱え、科学計算のコードで使いやすいテンソル層はまだ足りなかった。私たちが作りたかったのは、それらを置き換えるライブラリではない。既存のライブラリの間をつなぐスタックだ。

Rust の状況は、この数年で大きく変わった。crates.io は 2015 年の 602 crate から 2026 年には約 21 万 crate へ増えた（[データ](https://github.com/shinaoka/rust_crate_count)）。密な線形代数には faer があり、GPU カーネルには [CubeCL](https://github.com/tracel-ai/cubecl) があり、汎用数値には `num-traits` と `num-complex` がある。配列なら ndarray、線形代数なら nalgebra や faer、深層学習フレームワークなら Burn や candle、NumPy 風の配列 API なら numr もある。私たちが必要としていたのは、これらの間を埋める科学計算向けのテンソルスタックだった。列優先で、動的 shape を扱え、eager と traced の両方の自動微分を持ち、einsum や FFT、CPU/CUDA バックエンド、拡張可能な演算を備えたものだ。そこで tenferro-rs を作っている。faer と CubeCL の上に作り、再発明ではなく必要な部分を足していく。SparseIR.jl や Julia のテンソルネットワークコードを移植してみると、どの層が足りないのかも見えやすかった。

こうして [tenferro-rs](https://github.com/tensor4all/tenferro-rs) の開発が始まった。この記事では、このスタックを作っている理由と、コードを書くのが人間だけではなくなった今、なぜ Rust を選んだのかを解説する。

## なぜ今、Julia ではなく Rust なのか

2、3 年前なら、私は学生にまず Julia を勧めていただろう。Julia は数式に近い形で書け、メモリ管理も楽で、数値計算ライブラリも揃っていた。Rust は覚えることが多く、ライブラリもまだ不足していた。

今は同じことが言えない。Rust が変わったからではなく、コードを書くのが人間だけではなくなったからだ。

Fortran も Python も Julia も、人間が手でコードを書き、読み、保守するコストを下げる方向に発展してきた。可読性、REPL、数式に近い記法、入りやすさ。AI がコードを書くようになると、言語に求めるものも変わる。書く速さは前ほど問題ではない。学習コストも、多くの部分をエージェントが引き受ける。一方で、「数式のように読める」ことは正しさを保証してくれない。エイリアシング、ミューテーション、アロケーションは、行の見た目からは分からない。

私たちにとって、基準は「人間がどれだけ速く書けるか」から「正しさをどれだけ確かめられるか」に変わった。この基準で見ると、Rust を選ぶ理由がはっきりする。

具体的には、次の点が大きい。

- 所有権と型が、コンパイル時に広い範囲の誤りを排除する。`cargo check` は数秒で答えるので、エージェントが間違えても、実行時ではなくその場ですぐに分かる。
- ビルド、依存解決、テスト、ベンチマークが Cargo にまとまっている。CMake は不要、リンク時のバージョン衝突もない。スタック全体と依存をゼロからビルドしてもノート PC で 2 分程度、編集とテストのループは数十秒だ。
- Rust はモジュールと crate の階層に沿ってシンボルの可視性を制御する。エージェントは層の**内側**でしか動けない。別の crate の内部に手を伸ばしたり、抽象をこっそり壊したりはできない。AI が書いたコードベース（tenferro は約 13 万行）では、この制約が重要になる。
- ライフタイムや所有権の機械的な複雑さはエージェントに任せられるので、人間の注意はアルゴリズム、設計、正しさへ向かう。かつて Rust に不利に働いた序盤の学習コストは、いまは以前ほど問題にならない。

C++ や Python、Julia で大きなコードを書いていると、「この規模でまだ検証できるのか」という不安が出てくる。Rust では、明らかにその不安感が少なく感じる。

## 移植から、スタックへ

最初から汎用のテンソルライブラリを作ろうとしていたわけではない。必要なものを移植し、道具に悩まされる時間を減らしたかった。ただ、実装を進めるうちに、自動微分、バックエンド、演算の追加方法をテンソルネットワーク専用に閉じ込めると後で困ることが分かってきた。そこで、共通部分を独立したテンソルスタックとして設計することにした。

- 演算ファミリは、一枚岩のテンソル型の内側ではなく、それぞれの crate に分ける。
- 自動微分の規則は、テンソル型の外側に置く。Julia/ChainRules の教訓に従い、微分の規則は演算そのものに属するのであって、一つの具体的なテンソルクラスに属するのではない。AD の基盤である [tidu-rs](https://github.com/tensor4all/tidu-rs) は汎用で、テンソル型はその一利用者にすぎない。
- バックエンドとデバイスは明示する。CPU と GPU の間でデータが黙って移動することはない。ある演算をどのバックエンドで実行できるかと、その時点でどのデバイスが使えるかも分けて扱う。
- ストレージは列優先（column-major）にする。Fortran、Julia、MATLAB、LAPACK/BLAS と揃え、行優先のデータはストライド付きビューで扱えるようにする。不要な eager コピーは避ける。

この設計にしたことで、テンソルネットワーク以外の用途にも使える形になった。

## tenferro-rs を 2 分で

このスタックが提供するのは、型付きテンソル、`backward()` を伴う即時（eager）実行、`grad`/`vjp`/`jvp`/HVP を伴う traced グラフ、線形代数、einsum、FFT、そして明示的な CPU・CUDA（および実験的な WebGPU）バックエンドだ。

以下は PyTorch 流の eager 自動微分で、`sum(x²)`（勾配は `2x`）を、リポジトリの [`eager_autodiff_pytorch_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/eager_autodiff_pytorch_style.rs) からそのまま引用したものだ。

```rust
use tenferro_ad::{EagerRuntime, Tensor};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = EagerRuntime::new();
    let x = runtime.variable_from(Tensor::from_vec_col_major(
        vec![3],
        vec![1.0_f64, 2.0, 3.0],
    )?)?;

    let prediction = x.mul(&x).unwrap();
    let loss = prediction.reduce_sum(&[0])?;

    assert_eq!(loss.shape(), &[]);
    assert_close(loss.materialized()?.as_slice::<f64>().unwrap(), &[14.0]);

    loss.backward()?;

    let grad = x
        .grad()?
        .expect("tracked variable should receive a gradient");
    assert_eq!(grad.shape(), &[3]);
    assert_close(grad.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    x.clear_grad()?;
    assert!(x.grad()?.is_none());

    Ok(())
}
```

この例では、プログラムが自分で結果を確認している。検証を重視する方針が、この小さな例にも出ている。同じ計算は、一度コンパイルして再利用する JAX 流の traced グラフとしても、`grad`/`vjp`/`jvp` 付きで実行できる（[`traced_autodiff_jax_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/traced_autodiff_jax_style.rs) を参照）。どの層を使うかは選べる。型付きテンソル、自動微分付きの eager、あるいは traced グラフ。自動微分、CUDA、einsum、FFT、線形代数は、どれも必要なときだけ有効にできる。

## データでサイズが変わる計算

JAX と XLA は、shape が固定された計算をまとめて最適化し、同じ形の入力に対して高速に再実行するのが得意だ。ただし shape がデータに依存し始めると、途端に扱いづらくなる。打ち切り閾値、適応的なボンド次元、データ依存の反復回数。サイズが実行時にしか分からないと、新しい shape のたびに再コンパイルが必要になる。再コンパイルを避けて eager 実行に戻すと、`grad` や `vjp` などの変換を同じ流れでは使いにくくなる。

これはテンソルネットワークや、適応的な科学計算の多くにとって、日常の現実だ。私たちが扱うものの大半はそうだ。ランクやボンド次元がデータで変わる計算では、再コンパイルなしに同じ traced プログラムを使い続けられることが効いてくる。

tenferro は traced プログラムを一度だけコンパイルし、具体的なサイズ（ランク、閾値、反復回数）が実行時に決まっても、それを**再利用**する。その間も `grad`/`vjp`/`jvp` が使える。一方で、サイズが静的に決まる計算では OpenXLA を使えるようにしてある。`tenferro-xla` はグラフを StableHLO へ lower し、PJRT プラグインを読み込めるので、原理的には JAX と同じ計算速度に到達できる。

## 正しさを外から確かめる

AI が書いた数値計算ライブラリには懸念がある。見た目は正しそうでも、中身が間違っていないか。

そこで、信頼の根拠を「コードを読むこと」だけに置かない。目視に頼らず、別の仕組みで確かめられるようにする。

- 正しさは、有限差分と PyTorch の参照オラクルで検証する。[tensor-ad-oracles](https://github.com/tensor4all/tensor-ad-oracles) は、テンソルおよび線形代数演算の微分の正しさを調べるための、独立したデータベースと生成器だ。不変量、残差チェック、来歴（provenance）チェックも持たせている。
- 性能は、再現可能なスイート [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark) で、PyTorch や JAX と比較して測定する。多くのターゲットで、tenferro はすでに PyTorch や JAX に近い CPU/GPU 性能に達している。まだ最適化の途中で、伸びしろもある。ここでは数値を固定して書かず、ベンチマークのリポジトリにリンクする。そこが再現可能な参照元で、数値は更新されていくからだ。
- 設計も文書に残す。REPOSITORY_RULES.md、AGENTS.md、設計ノート、worklog にアーキテクチャと制約を書き、人間もエージェントも参照する。失敗から「演算ファミリは facade ではなく first-class crate にする」「素朴な CPU ループのフォールバックは禁止、faer か BLAS を使う」といった規則が見つかったら、その場限りの修正ではなく、新しい制約として記録する。13 万行のコードベースで設計がずれないようにするためだ。

オラクルもベンチマークも別々のリポジトリにあるので、ライブラリを書くエージェントが基準を動かすことはできない。規則とコードも継続的に突き合わせ、ずれがあれば早く見つける。さらに、ファイルごとのカバレッジ閾値を強制するテスト群を加える。生成されたコードをレビューするだけではなく、正しさ、性能、設計を別系統のチェックで確かめられるようにしておく。

## 試してみる

最初の tenferro crate 群は **crates.io** にある。単一の bundle ではなく、モジュラーなスタックそのものとして公開した。

```toml
[dependencies]
tenferro-tensor = "0.1"   # tensors, views, backends
tenferro-ad     = "0.1"   # eager + traced autodiff
# plus tenferro-linalg, tenferro-einsum, tenferro-fft, tenferro-cpu,
# tenferro-gpu, tenferro-xla: add only what you need
```

tenferro-rs はまだ v0.1 のプレビューで、1.0 として固まったものではない。一方で、単なる実験でもない。私たちの Rust テンソルネットワークスタック [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs)（TreeTN・QTT・TCI）のエンジンとして使い、実際の科学計算ワークロードで動かしながら開発している。ホスト言語が Python なら、JAX や PyTorch を使うのが自然だ。tenferro-rs は、同じような機能を Rust から直接使いたい人に向けて作っている。プレビュー段階なので、今なら利用者からのフィードバックを設計に反映しやすい。

特に、ndarray、nalgebra/faer、Burn/candle、PyTorch/JAX、あるいは Julia/Fortran を科学計算・HPC で使ってきた方に試してほしい。

- ドキュメントとガイド: https://tensor4all.org/tenferro-rs/
- ソース: https://github.com/tensor4all/tenferro-rs
- ベンチマークを再現する: https://github.com/tensor4all/tenferro-benchmark
- 正しさを試す: https://github.com/tensor4all/tensor-ad-oracles

実際の科学計算ワークロードで試して、足りないところ、遅いところ、間違っているところがあれば教えてほしい。そういう報告は非常に有益だ。

## 謝辞

初期の tenferro の設計について [Jin-Guo Liu](https://giggleliu.github.io/) さんに、開発支援について [Satoshi Terasaki](https://terasakisatoshi.github.io/) さんに感謝する。

*開示: この記事は AI コーディングエージェントと著者の協働で執筆され、記事が述べるのと同じ、人間が検証するワークフローで仕上げられた。*
