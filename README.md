# MiPN Negotiator

MiPN Negotiator は、複数人交渉ドメイン向けの強化学習エージェントです。
学習は主に `ppo_scratch.py` の独自 PPO 実装で行い、テストでは学習済み
`checkpoint.pt` を読み込んで、ルールベースの交渉相手と交渉させます。

## ディレクトリ構成

```text
.
|-- train.py                  # 学習の実行入口
|-- test_negotiator.py        # テスト / 評価の実行入口
|-- ppo_scratch.py            # MiPN 用 PPO 学習ループ
|-- policy.py                 # MiPN policy network
|-- rollout_buffer.py         # rollout buffer
|-- envs/                     # 交渉環境、ドメイン読み込み、モデル読み込み
|-- sao/                      # SAO mechanism と baseline negotiator
|-- domain/                   # GENIUS 形式の domain / utility XML
|-- run_command/              # 一括テスト用コマンド生成・実行スクリプト
|-- data_calculator/          # 結果集計スクリプト
|-- results/                  # checkpoint と評価結果
|-- data_SMIHT/, data_PHRI/   # 整理済み TSV 実験データ
`-- summary_tables/           # 集計テーブル出力
```

## 環境構築

推奨は Docker です。ホスト側のリポジトリを `/app` にマウントして使います。

```bash
docker build -t mipn:dev .
docker run --rm -it -v "$(pwd):/app" mipn:dev bash
```

コンテナに入った後は `/app` で以下のコマンドを実行します。

ローカル Python で実行する場合は Python 3.8 を使ってください。

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<24" "setuptools<66" "wheel<0.41"
python -m pip install -r requirements.txt
```

## 利用できる名前

主な学習用ドメイン:

```text
Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A
```

未学習ドメインでの汎化テストに使うドメイン:

```text
Coffee Camera Lunch SmartPhone Kitchen
```

利用できる交渉相手:

```text
Boulware Linear Conceder TitForTat1 TitForTat2 AgentK HardHeaded Atlas3 AgentGG
```

## 学習方法

MiPN だけを最小構成で学習する例です。

```bash
python3 ./train.py -a Boulware -i Laptop --skip_venas
```

`-a Boulware` のように相手を 1 つだけ指定すると、`Boulware-Boulware`
として学習します。出力先はデフォルトで次の形になります。

```text
results/<domain>_<agents>/<timestamp>-TA/MiPN_Negotiator/checkpoint.pt
```

例:

```text
results/Laptop_Boulware/20260627-010203-TA/MiPN_Negotiator/checkpoint.pt
```

相手ペアを明示して学習する例です。

```bash
python3 ./train.py -a Boulware Conceder -i Laptop --skip_venas
```

複数ドメイン・複数相手で general policy を学習する例です。

```bash
python3 ./train.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A \
  --skip_venas
```

動作確認用に短く学習する場合:

```bash
python3 ./train.py -a Boulware -i Laptop -t 8192 -n 4 -rs 2048 --skip_venas
```

出力先を指定する場合:

```bash
python3 ./train.py -a Boulware -i Laptop -sp ./results/debug_run/ --skip_venas
```

学習中のドメイン・相手ペア選択をランダムにする場合:

```bash
python3 ./train.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Laptop Car \
  --random_train \
  --skip_venas
```

MiPN と VeNAS baseline の両方を実行する場合は `--skip_venas` を外します。

```bash
python3 ./train.py -a Boulware -i Laptop
```

主な学習オプション:

```text
-a, --agents           交渉相手名。1 つだけ指定すると self-pair になる
-i, --issue            ドメイン名。複数指定可能
-sp, --save_path       出力先 root。デフォルトは ./results/
-t, --timesteps        学習 timesteps。デフォルトは 500000
-n, --n_envs           並列環境数。デフォルトは 4
-rs, --n_rollout_steps rollout 長。デフォルトは 2048
--skip_venas           VeNAS baseline の学習をスキップ
--random_train         ドメイン・相手ペアをランダム選択して学習
-gd, --general_domain  padding する observation / action サイズの基準ドメイン
```

## テスト方法

学習済みモデルのディレクトリを `-m` に指定してテストします。

```bash
python3 ./test_negotiator.py \
  -a Boulware \
  -i Laptop \
  -m ./results/Laptop_Boulware/20260627-010203-TA/MiPN_Negotiator/
```

明示的な相手ペアでテストする例:

```bash
python3 ./test_negotiator.py \
  -a Boulware Conceder \
  -i Laptop \
  -m ./results/Laptop_Boulware-Conceder/20260627-010203-TA/MiPN_Negotiator/
```

general policy を学習済みドメイン全体でテストする例:

```bash
python3 ./test_negotiator.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A \
  -m ./results/Laptop-ItexvsCypress-IS_BT_Acquisition-Grocery-thompson-Car-EnergySmall_A_Boulware-Conceder-Linear-Atlas3/20260627-010203-TA/MiPN_Negotiator/
```

general policy を未学習ドメインでテストする例:

```bash
python3 ./test_negotiator.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Coffee Camera Lunch SmartPhone Kitchen \
  -m ./results/Laptop-ItexvsCypress-IS_BT_Acquisition-Grocery-thompson-Car-EnergySmall_A_Boulware-Conceder-Linear-Atlas3/20260627-010203-TA/MiPN_Negotiator/
```

試しにエピソード数を減らす場合:

```bash
python3 ./test_negotiator.py \
  -a Boulware \
  -i Laptop \
  -m ./results/Laptop_Boulware/20260627-010203-TA/MiPN_Negotiator/ \
  -e 5
```

交渉過程の plot を保存する場合:

```bash
python3 ./test_negotiator.py \
  -a Boulware \
  -i Laptop \
  -m ./results/Laptop_Boulware/20260627-010203-TA/MiPN_Negotiator/ \
  -p
```

テスト結果はモデルディレクトリ配下に出力されます。

```text
<model_dir>/csv/<agent0>-<agent1>/<domain>/det=False_noise=False/*.tsv
<model_dir>/img/<agent0>-<agent1>/<domain>/det=False_noise=False/*.png
```

TSV の列:

```text
my_util  opp_util1  opp_util2  social  nash  agreement  step
```

## 一括テスト

`results/results_case5` のような results ディレクトリから、全 checkpoint
に対するテストコマンドを生成できます。

```bash
python3 ./run_command/command_generate.py results/results_case5 -o run_command/run_test5.sh
```

生成したスクリプトを実行します。

```bash
bash run_command/run_test5.sh
```

既存の `run_command/run_test1.sh` から `run_command/run_test6.sh` は一括実行用の例です。
結果ディレクトリを移動した後は、上の `command_generate.py` で再生成するのが安全です。

## 結果集計

整理済み TSV データを集計して、CSV とテキストテーブルを出力します。

```bash
python3 ./data_calculator/summary_data.py \
  --data-dir data_SMIHT \
  --output-dir summary_tables
```

未学習ドメインの結果を集計する場合:

```bash
python3 ./data_calculator/summary_unexpect_data.py \
  --data-dir data_SMIHT \
  --output-dir summary_tables
```

case 数や行数の不足をエラーとして扱う場合:

```bash
python3 ./data_calculator/summary_data.py --data-dir data_SMIHT --strict
```

集計スクリプトは次のような構造を想定しています。

```text
data_SMIHT/
|-- expert/<agent0>-<agent1>/<domain>/case1/*.tsv
`-- general/<agent0>-<agent1>/<domain>/case1/*.tsv
```

## 補足

- デフォルトの `--general_domain` は `EnergySmall_A` です。複数ドメインで扱えるように、observation と action はこのドメインを基準に padding されます。
- `test_negotiator.py` の `-m` には、基本的に `MiPN_Negotiator/` で終わるモデルディレクトリを渡してください。その中の `checkpoint.pt` が読み込まれます。
- Docker 実行時に Gym の deprecation warning が出ることがありますが、依存ライブラリ由来の警告であり、それだけで実行失敗を意味するものではありません。
