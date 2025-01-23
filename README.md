# route_search

Python で **ダイクストラ法** の進行をアニメーション表示し、最終的に確定した最短経路をハイライトするスクリプトを提供しています。GIF として保存することで可視化できます。

---

## 機能概要

- **辞書**を用いてグラフを定義し、NetworkX を使って可視化。
- ダイクストラ法を **ステップごと**に進め、各ステップの状態（確定したノードや暫定距離など）をアニメーションとして表示。
- CLI から **始点と終点を ID で入力**し、指定したノード間の **最短経路**を赤色でハイライト。
- 実行後、**GIF アニメーションが保存**されます。

---

## デモ

1. ノード一覧が表示されるので、始点と終点の ID を入力します。  
2. ダイクストラ法の進行状況をステップごとに描画したアニメーションが生成されます。  
3. 最終的な最短経路は赤色で描画されます。

（イメージサンプルです。実際の GIF は `./dijkstra_animation.gif` に保存されます。）

Sample

![dijkstra_animation](https://github.com/user-attachments/assets/62614a1f-e33e-4ba8-8e40-bc395c630da9)

---

## 動作環境

以下の環境で動作を確認しています。
- Python version: 3.10.12
- Platform: Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35
- matplotlib version: 3.8.2
- networkx version: 3.1
- PIL version: 10.2.0

---

### 実行例

```bash
$ python dijkstra_visualization.py

ノード一覧:
1: Tokyo
2: Nagoya
3: Osaka
4: Fukuoka
5: Sapporo

始点のIDを入力してください: 1
終点のIDを入力してください: 5
始点: Tokyo, 終点: Sapporo
アニメーションを 'dijkstra_animation.gif' として保存しました。
```

上記のように入力すると、`dijkstra_animation.gif` が生成され、Tokyo(1) から Sapporo(5) までの最短経路が可視化されます。

---

作者 GitHub: [@tomiokario](https://github.com/tomiokario) at METAPLUS
