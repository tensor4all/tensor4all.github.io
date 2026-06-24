---
title: tenferro-rs のご紹介
---

# Julia から Rust へ：エージェンティック AI 時代の、科学計算のための微分可能テンソルスタック

*[tenferro-rs](https://github.com/tensor4all/tenferro-rs) は Rust ネイティブな密テンソルスタックです。線形代数、PyTorch 流の eager 自動微分、JAX 流の traced 変換、NumPy 流の einsum、FFT、拡張可能な演算 crate 群、そして明示的な CPU/CUDA バックエンドを備えます。最初の crate 群は 2026 年 6 月 23 日（JST）に crates.io へ公開しました。*

文責: **Hiroshi Shinaoka**（埼玉大学）、tensor4all チーム

*🌐 [English](https://tensor4all.org/blog/introducing-tenferro-rs/) · [日本語](https://tensor4all.org/blog/introducing-tenferro-rs-ja/) · [简体中文](https://tensor4all.org/blog/introducing-tenferro-rs-zh/)*

![tenferro-rs architecture overview](tenferro-architecture.svg)

---

テンソルネットワークは Julia で書かれています。それには十分な理由があります。ITensors とその周辺エコシステムは優秀で、プロトタイピングは数式に近く書け、長年それはまさに正しい選択でした。私たち自身の研究（IR 基底、スパースモデリング、[tensor4all](https://tensor4all.org) のテンソルクロス補間（TCI）／quantics スタック）も、そこで育ってきました。

しかしコードベースが大きくなると、Julia での開発は具体的で見覚えのある形でつらくなってきます。実行時にしか表面化しない型不安定性、編集とテストのループを長くするコンパイル／プリコンパイル時間、そしてコードが育つほど正しさを抑え込みにくくなるという不安です。私たちはテンソルネットワークのスタックをより大きなものへ移植する過程でこの壁にぶつかり、エンジンを Rust へ移し始めました。

始めてすぐ、2 つめの驚きに出会いました。土台にしたいテンソルライブラリが、まだ揃っていなかったのです。Rust には強力な部品があります。配列の ndarray、深層学習の Burn、線形代数の faer。しかし、自動微分から einsum までを貫き、本格的な科学計算を支えられるエコシステムは、まだ未発達でした。目的は最初から、それらを置き換えることではありません。欠けている「補完するスタック」を作ることでした。

土台がどれだけ整ってきたかも言っておく価値があります。かつての反論はもう覆りました。crates.io は 2015 年の 602 crate から 2026 年には約 21 万 crate へと増え（[データ](https://github.com/shinaoka/rust_crate_count)）、基盤は堅牢です。密な線形代数の faer、GPU カーネルの [CubeCL](https://github.com/tracel-ai/cubecl)、汎用数値の `num-traits` と `num-complex`。個別の層としても部品は存在します。配列なら ndarray、線形代数なら nalgebra や faer、深層学習フレームワークなら Burn や candle、NumPy 流の配列 API なら numr。しかしそのどれも、列優先（column-major）で、動的 shape に対応し、eager と traced の両方の自動微分を備え、einsum・FFT・CPU/CUDA・拡張可能な演算を持ち、深層学習ではなく科学計算に向けたテンソルスタックではありません。その「真ん中の層」こそ、tenferro-rs が担うものです。faer と CubeCL の上に作り、再発明せず貢献し返します。欠けた部品の移植も、いまや安価です。SparseIR.jl でも Julia のテンソルネットワークスタックでも、私たちはそれをやってきました。

こうして生まれたのが [tenferro-rs](https://github.com/tensor4all/tenferro-rs) です。この記事は、なぜそれを作る価値があると考えるのか、そしてなぜ今、エージェンティック AI の時代に Rust なのか、についてです。

## 言語選択の基準が、ちょうど反転した

2023 年初頭、那覇のある晩に、私はこう言っていました。

> 「ノートブックでプロトタイプを書くと Julia は心地よい。コードが数式のように見え、レビューしやすく、メモリも面倒を見てくれる。Rust にも興味はあるけれど、普通の学生にはハードルが高いし、数値計算ライブラリもまだ揃っていないよね。」

そして 2026 年の私なら、こう言います。

> 「もうノートブックでプロトタイプは書かない。Rust の厳格で明示的なモデルは、エージェント的コーディングにはむしろ効率的だ。コードを書くのはエージェントだから、急な学習曲線は私の問題ではない。コードが多少長くなっても、数式との整合を保つのは難しくない。足りない数値計算ライブラリ？ 移植すればいい。」

変わったのは Rust ではありません。変わったのは、コードを書くのが誰か、です。

Fortran も Python も Julia も、ある一つのことを安くするよう設計されていました。人間が手でコードを書き、読み、保守するコストです。可読性、REPL、数式に近い記法、緩やかな入り口。AI がコードを書くようになると、これらの利点は力を失います。書く速さはボトルネックでなくなり、急な学習曲線はエージェントが肩代わりし、「数式のように読める」ことはもはや正しさを保証しません。エイリアシング、ミューテーション、アロケーションは、行の見た目からは見えないからです。

だから言語選択の基準は反転します。コードの読みやすさから、検証しやすさへ。可読性から、検査可能性（inspectability）へ。この捉え直しこそ、tenferro-rs が Rust である理由のすべてです。

## 移植から、「スタック」へ

汎用のテンソルライブラリを作ろうとしていたわけではありません。必要なものを移植し、道具と戦うのをやめようとしただけです。しかし早い段階で打ったいくつかの設計上の賭けが、テンソルネットワークの移植を、より広いものへと静かに変えていきました。

- **モノリスではなく、モジュラーな crate 群。** 演算ファミリは、一枚岩のテンソル型の内側ではなく、それぞれの crate に住みます。
- **自動微分の規則は、テンソル型の外側に置く。** Julia/ChainRules の教訓に従い、微分の規則は演算の意味論に属するのであって、一つの具体的なテンソルクラスに属するのではありません。AD の基盤（[tidu-rs](https://github.com/tensor4all/tidu-rs)、[chainrules-rs](https://github.com/tensor4all/chainrules-rs)）は汎用で、テンソル型はその一利用者にすぎません。
- **バックエンドとデバイスは明示的に。** CPU と GPU の間でデータが黙って移動することはありません。能力と、実行時の可用性は、別の関心事です。
- **列優先（column-major）ストレージ。** Fortran、Julia、MATLAB、LAPACK/BLAS と揃い、ストライド付きビューで行優先のデータへも、eager なコピーなしに橋渡しします。

これらの選択ゆえに、結果はテンソルネットワークを超えて使えるものになりました。

## tenferro-rs を 2 分で

このスタックが提供するのは、型付きテンソル、`backward()` を伴う即時（eager）実行、`grad`/`vjp`/`jvp`/HVP を伴う traced グラフ、線形代数、einsum、FFT、そして明示的な CPU・CUDA（および実験的な WebGPU）バックエンドです。

以下は PyTorch 流の eager 自動微分で、`sum(x²)`（勾配は `2x`）を、リポジトリの [`eager_autodiff_pytorch_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/eager_autodiff_pytorch_style.rs) からそのまま引用したものです。

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

アサーションは、プログラムが自分自身を検証している様子です。検証を旨とするライブラリにふさわしい姿です。同じ計算は、一度コンパイルして再利用する JAX 流の traced グラフとしても、`grad`/`vjp`/`jvp` 付きで走ります（[`traced_autodiff_jax_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/traced_autodiff_jax_style.rs) を参照）。問題を解く最小の層を選べます。型付きテンソル、自動微分付きの eager、あるいは traced グラフ。自動微分・CUDA・einsum・FFT・線形代数は、すべてオプトインです。

## 他所では本当に難しいところ: 実行時まで分からない shape

JAX と XLA は素晴らしい。shape がデータに依存し始めるまでは。打ち切り閾値、適応的なボンド次元、データ依存の反復回数。サイズが実行時にしか分からなくなった瞬間、新しい shape のたびに再コンパイルするか、eager に退避して変換（transforms）を失うことになります。

これはテンソルネットワークや、適応的な科学計算の多くにとって、日常の現実です。私たちが生きている場所です。

tenferro は traced プログラムを一度だけコンパイルし、具体的なサイズ（ランク、閾値、反復回数）が実行時に解決される間、それを**再利用**します。その全体を通して `grad`/`vjp`/`jvp` が使え、しかも純 Rust です。静的 shape は前提ではなく、特殊ケースです。（静的 shape のグラフについては、`tenferro-xla` が StableHLO へ lower し、PJRT プラグインを読み込むこともできます。）

shape が常に固定なら、これは響かないでしょう。そうでないなら、私たちが最初に指し示す機能です。

## なぜ、エージェンティック AI 時代に Rust なのか（具体的に）

テーゼを超えて、日々の理由も成り立ちます。

- **間違いは実行前に捕まる。** 所有権と型が、コンパイル時に広いクラスの誤りを排除します。`cargo check` は数秒で答えるので、エージェントが間違えても、実行時ではなくその場ですぐに分かります。
- **Cargo がビルドシステム。** ビルド、依存解決、テスト、ベンチマークが一つの道具にまとまっています。CMake は不要、リンク時のバージョン衝突もありません。スタック全体と依存をゼロからビルドしてもノート PC で 2 分程度、編集とテストのループは数十秒です。
- **crate 境界は、衛生ではなく強制。** Rust はモジュールと crate の階層に沿ってシンボルの可視性を制御するので、エージェントは層の**内側**でしか動けません。別の crate の内部に手を伸ばしたり、抽象をこっそり壊したりはできません。AI が書いたコードベース（tenferro は約 13 万行）では、この構造的な制約こそが、複雑さの増大を食い止めます。
- **難しい部分はエージェントが吸収する。** ライフタイムや所有権の機械的な複雑さ（Rust の悪名高い学習の崖）はエージェントが扱うので、人間の注意はアルゴリズム、設計、正しさへ向かいます。かつて Rust に不利に働いた急な序盤の勾配は、いまや、気にしない相手が払ってくれます。

C++ や Python、Julia で多くの人が身につけた不安、すなわち「コードが大きくなると検証が難しくなる」という不安は、消えてなくなります。

## checked であって、vibe ではない

AI が書いた数値計算ライブラリへの当然の懸念は、すべての AI コードへの当然の懸念と同じです。もっともらしいだけの「slop（がらくた）」ではないのか？

私たちの答えは、信頼の根拠を「コードを読むこと」から、誰の目視にも依存しない仕組みへ移すことです。

- **正しさ**は、有限差分と PyTorch の参照オラクルに照らして検証します。[tensor-ad-oracles](https://github.com/tensor4all/tensor-ad-oracles) は、テンソルおよび線形代数演算の微分の正しさのための、独立したデータベースと生成器であり、不変量・残差チェック・来歴（provenance）チェックに支えられています。
- **性能**は、再現可能なスイート [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark) で、PyTorch や JAX と比較して測ります。多くのターゲットで、tenferro はすでに CPU 性能で同等に達しており、なお最適化中で、伸びしろも残っています。数値をここに焼き込まず、ベンチマークのリポジトリにリンクするのは、そこが正本（canonical）であり、再現可能で、数値は動くからです。
- **設計の一貫性**は、誰かの頭の中ではなく、文書に書き出します。育っていく source of truth（REPOSITORY_RULES.md、AGENTS.md、設計ノート、worklog）が、アーキテクチャとその制約を記録し、人間もエージェントもそれに従います。失敗が、欠けていた規則をあぶり出したとき（たとえば「演算ファミリは facade ではなく first-class crate にする」「素朴な CPU ループのフォールバックは禁止、faer か BLAS を使う」）、それは場当たりのパッチではなく、新しい「書かれた制約」になります。13 万行のコードベースが、セッションごとに漂流せず一つの統一された設計を保てるのは、こうしてです。

オラクルもベンチマークも別々のリポジトリにあるので、ライブラリを書くエージェントがゴールポストを動かすことはできません。規則はコードと継続的に突き合わされ、両者が静かに食い違うこともありません。ここにファイルごとのカバレッジ閾値を強制する網羅的なテスト群を加えれば、出来上がるのは vibe coding の正反対の開発スタイルです。正しさも、性能も、設計も、目分量ではなく、外部から体系的に確立されます。

## 試して、形づくる側に

最初の tenferro crate 群は本日、**crates.io** にあります。単一の bundle ではなく、モジュラーなスタックそのものとして公開しました。

```toml
[dependencies]
tenferro-tensor = "0.1"   # tensors, views, backends
tenferro-ad     = "0.1"   # eager + traced autodiff
# plus tenferro-linalg, tenferro-einsum, tenferro-fft, tenferro-cpu,
# tenferro-gpu, tenferro-xla: add only what you need
```

まだ早い段階です。v0.1 であり、安定した 1.0 ではなくプレビューです。ですが、おもちゃではありません。tenferro-rs はすでに、私たちの Rust テンソルネットワークスタック [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs)（TreeTN・QTT・TCI）のエンジンとして、開発の中で実際の科学計算ワークロードにさらされています。ホスト言語が Python なら、JAX や PyTorch が自然な選択です。tenferro-rs は、この種のスタックを Rust ネイティブで欲しいプロジェクトのためのものです。そして、あなたのフィードバックで行き先が変わるくらい、早い段階にあります。

特に、ndarray、nalgebra/faer、Burn/candle、PyTorch/JAX、あるいは Julia/Fortran を科学計算・HPC で使ってきた方の声を聞きたいです。

- **ドキュメントとガイド:** https://tensor4all.org/tenferro-rs/
- **ソース:** https://github.com/tensor4all/tenferro-rs
- **ベンチマークを再現する:** https://github.com/tensor4all/tenferro-benchmark
- **正しさを試す:** https://github.com/tensor4all/tensor-ad-oracles

実際の科学計算ワークロードで試して、足りない・遅い・間違っているところがあれば、それこそ私たちが欲しい報告です。

## 謝辞

初期の tenferro の設計について [Jin-Guo Liu](https://giggleliu.github.io/) さんに、開発支援について [Satoshi Terasaki](https://terasakisatoshi.github.io/) さんに感謝します。

*開示: この記事は AI コーディングエージェントと著者の協働で執筆され、記事が述べるのと同じ、人間が検証するワークフローで仕上げられました。*
