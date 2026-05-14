# Transformer-based Negotiation AI Agent
## Overview
implementation code for VeNAS and MiPN -based negotiation AI agent architecture
## Instructions

### Docker

`mipn:dev` イメージは以下でビルドできます。

```bash
docker build -t mipn:dev .
```

対話的にコンテナへ入る場合:

```bash
docker run --rm -it -v "$(pwd):/app" mipn:dev bash
```

学習をそのまま実行する場合:

```bash
docker run --rm -it -v "$(pwd):/app" mipn:dev python3 ./train.py -a Boulware -i Laptop
```

### Requirements

```
absl-py==0.13.0
cachetools==4.2.2
certifi==2021.5.30
charset-normalizer==2.0.5
click==8.0.1
click-config-file==0.6.0
cloudpickle==1.6.0
colorlog==6.4.1
configobj==5.0.6
cycler==0.10.0
dill==0.3.4
future==0.18.2
gif==3.0.0
google-auth==1.35.0
google-auth-oauthlib==0.4.6
grpcio==1.40.0
gym==0.26.2
idna==3.2
importlib-metadata==4.8.1
inflect==5.3.0
joblib==1.0.1
kiwisolver==1.3.2
Markdown==3.3.4
matplotlib==3.4.3
negmas==0.8.8
networkx==2.6.2
numpy==1.21.2
oauthlib==3.1.1
pandas==1.3.2
Pillow==7.2.0
progressbar2==3.53.2
protobuf==3.17.3
psutil==5.8.0
py4j==0.10.9.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyglet==1.5.0
pyparsing==2.4.7
pytest-runner==5.3.1
python-dateutil==2.8.2
python-utils==2.5.6
pytz==2021.1
PyYAML==5.4.1
requests==2.26.0
requests-oauthlib==1.3.0
rsa==4.7.2
scikit-learn==0.24.2
scipy==1.7.1
seaborn==0.11.2
six==1.16.0
sklearn==0.0
stable-baselines3==1.2.0
stringcase==1.2.0
tabulate==0.8.9
tensorboard==2.6.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
threadpoolctl==2.2.0
torch==1.9.0
tqdm==4.62.2
typing==3.7.4.3
typing-extensions==3.10.0.2
urllib3==1.26.6
Werkzeug==2.0.1
zipp==3.5.0

```


### Running experiments

- Example command for training:
```
python3 ./train.py -a Boulware Conceder Linear TitForTat1 TitForTat2 -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A
```
python3 ./train.py -a Boulware -i Laptop 

MiPN_Negotiator checkpoints are saved under `MiPN_Negotiator/checkpoint.pt`. Multiple domains can be trained together; observations are zero-padded to the general domain capacity and invalid issue/value actions are masked during PPO updates. By default, `--general_domain EnergySmall_A` is used, so even `-i Laptop` training creates the same input/action size needed by `Car` and `EnergySmall_A`.

- Example command for testing with a pretrained model:
```
python3 ./test_negotiator.py -a Boulware Conceder Linear TitForTat1 TitForTat2 -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A -m ./results/260311-034347/MiPN
```

python3 ./test_negotiator.py -a Boulware -i Laptop -m ./results/Laptop_Boulware-Boulware/20260320-054646-TA/MiPN/
python3 ./test_negotiator.py -a Boulware -i Laptop -m ./results/Laptop_Boulware-Boulware/MiPN/

- `-a` and `-i` arguments specify agents and issues for training or testing

## 各ファイル・クラスの説明
### ./train.py
- 学習実行
### ./test_negotiator.py
- テスト実行
### ./ppo_scratch.py
- 学習アルゴリズム実装部
- `PPO` : 環境を切り替えながら学習ループを回す．ロールアウトを実行し集めたデータで勾配更新の流れ
- stable-baselines3の実装をベースにAI agent用に改良（複数環境の切り替えやTransformer用の拡張）
### ./policy.py
- 各コンポーネントをまとめて全体のモデルにしている部分．ロールアウトバッファに関する定義もここにある
- `Transformer_Policy` : モデル本体．Transformer，方策・価値ネットワークなどを備え順伝搬等の処理を記述してある
- `RolloutBuffer` : シミュレーション時のデータを格納するためのバッファの定義&GAEの計算処理もここにある
### ./NegTransformer.py
- AIエージェントに用いたTransformer部分の実装．
### ./envs/env.py
- gymnasium環境を継承した交渉シミュレーション用環境を定義．
- `NaiveEnv` : RL用の各種変数やドメイン読み込み，交渉セッション，報酬等の定義
- `AOPEnv` : `step`を改良しAOP準拠の挙動を実装
### ./envs/rl_negotiator.py
- 学習時・テスト時のAIエージェント本体であるNegotiatorを実装
- `RLNegotiator` : 学習時のエージェント本体．`env.py`の`step`時に選択された`self.next_bid`がそのまま相手に提案される
- `TestRLNegotiator.py` : テスト時のエージェント本体．チェックポイントからモデルをロードしそれを用いて推論を行う．
### ./envs/observer.py
- bid履歴を観測するためのobserverを定義
- `EmbeddedObserveHistroy` : OpenAIのテキスト埋め込みによる埋込ベクトルをjsonファイルからロードし，実際のbidを埋め込みバッファに格納する
### ./embedding_model.py
- 新規に埋め込みベクトルを作成したい場合はこのファイルを実行．
- `MyEmbedding` : 埋め込みベクトルを新規作成 or 作成済みの埋め込みベクトルを`embeddings`に保存してあるjsonファイルからロードする
- `self.client = openai.OpenAI(api_key='')`にapi_keyを入力
