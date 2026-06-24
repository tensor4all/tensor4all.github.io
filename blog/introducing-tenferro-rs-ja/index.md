---
title: tenferro-rs のご紹介
---

# Julia から Rust へ：AI エージェントがコードを書く時代の、科学計算向け微分可能テンソルスタック

*[tenferro-rs](https://github.com/tensor4all/tenferro-rs) は Rust ネイティブな密テンソルスタックだ。線形代数、PyTorch 流の eager 自動微分、JAX 流の traced 変換、NumPy 流の einsum、FFT、拡張可能な演算 crate 群、そして明示的な CPU/CUDA バックエンドを備える。最初の crate 群は 2026 年 6 月 23 日（JST）に crates.io へ公開した。*

文責: **Hiroshi Shinaoka**（埼玉大学）、tensor4all チーム

*🌐 [English](https://tensor4all.org/blog/introducing-tenferro-rs/) · [日本語](https://tensor4all.org/blog/introducing-tenferro-rs-ja/) · [简体中文](https://tensor4all.org/blog/introducing-tenferro-rs-zh/)*

![tenferro-rs architecture overview](tenferro-architecture.svg)

---

テンソルネットワークのコードはだいたい Julia で書かれており、私たちのものも例外ではなかった。ITensors とその周辺エコシステムは、数式に近い形でプロトタイプを書ける。速く試行錯誤するときに都合が良い。IR 基底、スパースモデリング、そして [tensor4all](https://tensor4all.org) のテンソルクロス補間（TCI）／quantics スタックも、そこで育ててきた。

しかしコードベースが大きくなると、Julia での開発は無視できない形でつらくなる。実行時にしか表面化しない型不安定性、編集とテストのループを長くするコンパイル／プリコンパイル時間、そしてコードが育つほど正しさを抑え込みにくくなる不安だ。テンソルネットワークのスタックをより大きなシステムへ移植しているうちに、これらの問題が限界に達し、エンジンを Rust へ移し始めた。

始めてすぐ、2 つめの問題に出会った。土台にしたいテンソルライブラリが、まだ揃っていなかったのだ。Rust には強力な部品がある。配列の ndarray、深層学習の Burn、線形代数の faer。しかし、自動微分から einsum までを貫き、本格的な科学計算を支えられるエコシステムは、まだ未発達だった。目的は最初から、それらを置き換えることではない。欠けている補完的なスタックを作ることだ。

土台がどれだけ整ってきたかも言っておく価値がある。かつての反論はもう覆った。crates.io は 2015 年の 602 crate から 2026 年には約 21 万 crate へと増え（[データ](https://github.com/shinaoka/rust_crate_count)）、基盤は堅牢だ。密な線形代数の faer、GPU カーネルの [CubeCL](https://github.com/tracel-ai/cubecl)、汎用数値の `num-traits` と `num-complex`。個別の層としても部品は存在する。配列なら ndarray、線形代数なら nalgebra や faer、深層学習フレームワークなら Burn や candle、NumPy 流の配列 API なら numr。しかしそのどれも、私たちが必要としているものではなかった。列優先で動的 shape に対応し、eager と traced の両方の自動微分を持ち、einsum や FFT、CPU/CUDA バックエンド、拡張可能な演算を備え、かつ科学計算を想定したテンソルスタックが欠けていたのだ。その真ん中の層こそ、tenferro-rs が担うものだ。faer と CubeCL の上に作り、再発明せず貢献し返す。欠けた部品の移植も、いまや安価だ。SparseIR.jl でも Julia のテンソルネットワークスタックでも、私たちはそれをやってきた。

こうして生まれたのが [tenferro-rs](https://github.com/tensor4all/tenferro-rs) だ。この記事は、なぜそれを作る価値があると考えるのか、そしてなぜ今、コードを書くのが人間だけでなくなった時代に Rust なのか、について書く。

## なぜ今、Julia ではなく Rust なのか

2、3 年前なら、私は学生に「まず Julia から始めろ」と言っていた。Julia は数式のように書け、メモリ管理も楽だし、数値計算ライブラリも揃っていた。Rust は覚えることが多く、ライブラリもまだ不足していた。

今は同じことが言えない。Rust が変わったからではなく、コードを書くのが人間だけではなくなったからだ。

Fortran も Python も Julia も、ある一つのことを安くするよう設計されていた。人間が手でコードを書き、読み、保守するコストだ。可読性、REPL、数式に近い記法、緩やかな入り口。AI がコードを書くようになると、これらの利点は力を失う。書く速さはボトルネックでなくなり、学習コストはエージェントが肩代わりし、「数式のように読める」ことはもはや正しさを保証しない。エイリアシング、ミューテーション、アロケーションは、行の見た目からは見えないからだ。

私たちにとって、基準は「人間がどれだけ速く書けるか」から「正しさをどれだけ確かめられるか」に変わった。この捉え直しこそ、tenferro-rs が Rust である理由だ。

## 移植から、スタックへ

汎用のテンソルライブラリを作ろうとしていたわけではない。必要なものを移植し、道具と戦うのをやめようとしただけだ。しかし早い段階で打ったいくつかの設計上の賭けが、テンソルネットワークの移植を、より広いものへと変えていった。

- **モノリスではなく、モジュラーな crate 群。** 演算ファミリは、一枚岩のテンソル型の内側ではなく、それぞれの crate に住む。
- **自動微分の規則は、テンソル型の外側に置く。** Julia/ChainRules の教訓に従い、微分の規則は演算の意味論に属するのであって、一つの具体的なテンソルクラスに属するのではない。AD の基盤（[tidu-rs](https://github.com/tensor4all/tidu-rs)、[chainrules-rs](https://github.com/tensor4all/chainrules-rs)）は汎用で、テンソル型はその一利用者にすぎない。
- **バックエンドとデバイスは明示的に。** CPU と GPU の間でデータが黙って移動することはない。能力と、実行時の可用性は、別の関心事だ。
- **列優先（column-major）ストレージ。** Fortran、Julia、MATLAB、LAPACK/BLAS と揃い、ストライド付きビューで行優先のデータへも、eager なコピーなしに橋渡しする。

これらの選択ゆえに、結果はテンソルネットワークを超えて使えるものになった。

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

アサーションは、プログラムが自分自身を検証している様子だ。検証を旨とするライブラリにふさわしい。同じ計算は、一度コンパイルして再利用する JAX 流の traced グラフとしても、`grad`/`vjp`/`jvp` 付きで走る（[`traced_autodiff_jax_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/traced_autodiff_jax_style.rs) を参照）。問題を解く最小の層を選べる。型付きテンソル、自動微分付きの eager、あるいは traced グラフ。自動微分、CUDA、einsum、FFT、線形代数は、すべてオプトインだ。

## 他所では本当に難しいところ: 実行時まで分からない shape

JAX と XLA は素晴らしい。shape がデータに依存し始めるまでは。打ち切り閾値、適応的なボンド次元、データ依存の反復回数。サイズが実行時にしか分からなくなった瞬間、新しい shape のたびに再コンパイルするか、eager に退避して変換（transforms）を失うことになる。

これはテンソルネットワークや、適応的な科学計算の多くにとって、日常の現実だ。私たちが扱うものの大半はそうだ。

tenferro は traced プログラムを一度だけコンパイルし、具体的なサイズ（ランク、閾値、反復回数）が実行時に解決される間、それを**再利用**する。その全体を通して `grad`/`vjp`/`jvp` が使え、しかも純 Rust だ。静的 shape は前提ではなく、特殊ケースだ。（静的 shape のグラフについては、`tenferro-xla` が StableHLO へ lower し、PJRT プラグインを読み込むこともできる。）

shape が常に固定なら、それほど恩恵を受けないだろう。そうでないなら、私たちが最初に指し示す機能だ。

## なぜ、今 Rust なのか（具体的に）

大げさな話はさておき、日々の理由も成り立つ。

- **間違いは実行前に捕まる。** 所有権と型が、コンパイル時に広いクラスの誤りを排除する。`cargo check` は数秒で答えるので、エージェントが間違えても、実行時ではなくその場ですぐに分かる。
- **Cargo がビルドシステム。** ビルド、依存解決、テスト、ベンチマークが一つの道具にまとまっている。CMake は不要、リンク時のバージョン衝突もない。スタック全体と依存をゼロからビルドしてもノート PC で 2 分程度、編集とテストのループは数十秒だ。
- **crate 境界は、衛生ではなく強制。** Rust はモジュールと crate の階層に沿ってシンボルの可視性を制御するので、エージェントは層の**内側**でしか動けない。別の crate の内部に手を伸ばしたり、抽象をこっそり壊したりはできない。AI が書いたコードベース（tenferro は約 13 万行）では、この構造的な制約こそが、複雑さの増大を食い止める。
- **難しい部分はエージェントが吸収する。** ライフタイムや所有権の機械的な複雑さはエージェントが扱うので、人間の注意はアルゴリズム、設計、正しさへ向かう。かつて Rust に不利に働いた序盤の学習コストは、いまや気にしない相手が払ってくれる。

C++ や Python、Julia で多くの人が身につけた不安、すなわち「コードが大きくなると検証が難しくなる」という不安は、小さくなる。

## 検証してから信じる

AI が書いた数値計算ライブラリへの当然の懸念は、他の AI 書きコードへの懸念と同じだ。見た目は正しくても、中身が間違っていないか。

私たちの答えは、信頼の根拠を「コードを読むこと」から、誰の目視にも依存しない仕組みへ移すことだ。

- **正しさ.** 有限差分と PyTorch の参照オラクルで検証する。[tensor-ad-oracles](https://github.com/tensor4all/tensor-ad-oracles) は、テンソルおよび線形代数演算の微分の正しさのための、独立したデータベースと生成器であり、不変量・残差チェック・来歴（provenance）チェックに支えられている。
- **性能.** 再現可能なスイート [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark) で、PyTorch や JAX と比較して測定する。多くのターゲットで、tenferro はすでに CPU 性能で同等に達しており、なお最適化中で、伸びしろも残っている。数値をここに焼き込まず、ベンチマークのリポジトリにリンクするのは、そこが正確な参照源であり、再現可能で、数値は動くからだ。
- **設計の一貫性.** 誰かの頭の中に置くのではなく、文書に残す。育っていく source of truth（REPOSITORY_RULES.md、AGENTS.md、設計ノート、worklog）が、アーキテクチャとその制約を記録し、人間もエージェントもそれに従う。失敗が、欠けていた規則をあぶり出したとき（たとえば「演算ファミリは facade ではなく first-class crate にする」「素朴な CPU ループのフォールバックは禁止、faer か BLAS を使う」）、それは場当たりのパッチではなく、新しい「書かれた制約」になる。13 万行のコードベースが、セッションごとに漂流せず一つの統一された設計を保てるのは、こうしてだ。

オラクルもベンチマークも別々のリポジトリにあるので、ライブラリを書くエージェントがゴールポストを動かすことはできない。私たちは規則をコードと継続的に突き合わせるので、両者が静かに食い違うこともない。ここにファイルごとのカバレッジ閾値を強制する網羅的なテスト群を加えれば、出来上がるのは、vibe coding とは全く違う開発スタイルだ。正しさも、性能も、設計も、目分量ではなく、外部から体系的に確立する。

## 試してみる

最初の tenferro crate 群は本日、**crates.io** にある。単一の bundle ではなく、モジュラーなスタックそのものとして公開した。

```toml
[dependencies]
tenferro-tensor = "0.1"   # tensors, views, backends
tenferro-ad     = "0.1"   # eager + traced autodiff
# plus tenferro-linalg, tenferro-einsum, tenferro-fft, tenferro-cpu,
# tenferro-gpu, tenferro-xla: add only what you need
```

まだ早い段階だ。v0.1 のプレビューだ。だが、単なる実験ではない。tenferro-rs はすでに、私たちの Rust テンソルネットワークスタック [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs)（TreeTN・QTT・TCI）のエンジンとして、開発の中で実際の科学計算ワークロードにさらされている。ホスト言語が Python なら、JAX や PyTorch が自然な選択だ。tenferro-rs は、この種のスタックを Rust ネイティブで欲しいプロジェクトのためのものだ。そして、あなたのフィードバックで行き先が変わるくらい、早い段階にある。

特に、ndarray、nalgebra/faer、Burn/candle、PyTorch/JAX、あるいは Julia/Fortran を科学計算・HPC で使ってきた方の声を聞きたい。

- **ドキュメントとガイド:** https://tensor4all.org/tenferro-rs/
- **ソース:** https://github.com/tensor4all/tenferro-rs
- **ベンチマークを再現する:** https://github.com/tensor4all/tenferro-benchmark
- **正しさを試す:** https://github.com/tensor4all/tensor-ad-oracles

実際の科学計算ワークロードで試して、足りない・遅い・間違っているところがあれば、それが私たちにとって一番役立つ報告だ。

## 謝辞

初期の tenferro の設計について [Jin-Guo Liu](https://giggleliu.github.io/) さんに、開発支援について [Satoshi Terasaki](https://terasakisatoshi.github.io/) さんに感謝する。

*開示: この記事は AI コーディングエージェントと著者の協働で執筆され、記事が述べるのと同じ、人間が検証するワークフローで仕上げられた。*
