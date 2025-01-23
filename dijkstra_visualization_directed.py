import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import sys

# ----------------------------
# グラフ構築（隣接リスト形式の辞書から）
# ----------------------------
def create_graph_from_dict(adj_dict):
    """
    有向グラフを構築する。
    """
    # ★ここを DiGraph にするのがポイント
    G = nx.DiGraph()
    for src, neighbors in adj_dict.items():
        G.add_node(src)
        for dst, w in neighbors.items():
            G.add_node(dst)
            # 有向辺を追加
            G.add_edge(src, dst, weight=w)
    pos = nx.spring_layout(G, seed=42)
    return G, pos

# ----------------------------
# ダイクストラ法（ステップごとの状態を yield）
# ----------------------------
def dijkstra_steps(G, start):
    """
    ダイクストラ法を1ステップずつ進め、その都度の状態を yield する。
    戻り値 (各ステップ) のタプル:
        (
            visited,         # 確定したノード集合
            dist,            # 各ノードへの暫定距離
            current_node,    # 新しく確定したノード
            neighbor_node,   # 更新が起きた隣接ノード
            prev,            # 前任ノードの辞書
        )
    """
    dist = {node: float('inf') for node in G.nodes}
    dist[start] = 0
    prev = {node: None for node in G.nodes}

    unvisited = set(G.nodes)
    visited = set()

    while unvisited:
        # 未確定ノードの中で暫定距離が最小のノードを取得
        current_node = min(unvisited, key=lambda node: dist[node])
        if dist[current_node] == float('inf'):
            break

        unvisited.remove(current_node)
        visited.add(current_node)

        # current_node が確定した直後
        yield visited.copy(), dist.copy(), current_node, None, prev.copy()

        # current_node から「有向辺」でつながる隣接ノードだけを更新
        for neighbor in G[current_node]:
            if neighbor in unvisited:
                edge_weight = G[current_node][neighbor]['weight']
                new_distance = dist[current_node] + edge_weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    prev[neighbor] = current_node
                    yield visited.copy(), dist.copy(), current_node, neighbor, prev.copy()

# ----------------------------
# 最短パス復元用関数
# ----------------------------
def reconstruct_path(prev, start, end):
    """
    prevを辿って start->end の経路を edge のリストとして返す。
    """
    path_edges = []
    node = end
    while node is not None and node != start:
        parent = prev[node]
        if parent is None:
            break
        path_edges.append((parent, node))
        node = parent
    path_edges.reverse()
    return path_edges

# ----------------------------
# 描画用 (アニメーション)
# ----------------------------
def visualize_dijkstra(G, pos, start_node, end_node=None):
    """
    ダイクストラ法の進行をアニメーション表示。
    end_node が指定された場合、start->end の最短経路を赤色でハイライト。
    """
    steps = list(dijkstra_steps(G, start_node))

    fig, ax = plt.subplots(figsize=(8, 6))

    def draw_graph(visited, dist, current, neighbor, prev):
        ax.clear()
        ax.set_title(f"Dijkstra's Algorithm (Start: {start_node}, End: {end_node})", fontsize=14)

        # ----------------------------------------------------------
        #  1) 両方向エッジがある場合は forward_edges / backward_edges を振り分ける
        # ----------------------------------------------------------
        forward_edges = []
        backward_edges = []

        # 文字列としての比較で u < v なら「片方」など、何らかの規則で分離
        # （ノード名が同じケースは通常想定しない）
        for (u, v) in G.edges():
            if G.has_edge(v, u) and u < v:
                # 片方向として追加 (u->v) は forward に
                forward_edges.append((u, v))
            elif G.has_edge(v, u) and u > v:
                # (u->v) は backward に
                backward_edges.append((u, v))
            else:
                # 一方向しかない場合は forward に入れておく
                forward_edges.append((u, v))

        # ----------------------------------------------------------
        #  2) 分けたエッジをそれぞれ「曲率を変えて」描画
        # ----------------------------------------------------------
        # forward_edges は arc3, rad=+0.2
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=forward_edges,
            edge_color='blue',
            alpha=0.5,
            arrows=True,
            arrowstyle='->',
            connectionstyle='arc3, rad=0.2'
        )
        # backward_edges は arc3, rad=0.2
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=backward_edges,
            edge_color='green',
            alpha=0.5,
            arrows=True,
            arrowstyle='->',
            connectionstyle='arc3, rad=0.2'
        )

        # ----------------------------------------------------------
        #  3) エッジラベル（重み）をそれぞれずらして表示
        # ----------------------------------------------------------
        edge_labels_all = nx.get_edge_attributes(G, 'weight')

        # forward 用ラベル
        forward_labels = {(u, v): edge_labels_all[(u, v)]
                          for (u, v) in forward_edges if (u, v) in edge_labels_all}
        # backward 用ラベル
        backward_labels = {(u, v): edge_labels_all[(u, v)]
                           for (u, v) in backward_edges if (u, v) in edge_labels_all}

        # forward 側は label_pos=0.3 ぐらい
        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=forward_labels,
            font_color='blue',
            label_pos=0.6,
            rotate=False
        )

        # backward 側は label_pos=0.7 ぐらい
        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=backward_labels,
            font_color='green',
            label_pos=0.6,
            rotate=False
        )

        # ----------------------------------------------------------
        #  4) 残りのノード、暫定距離、確定ノードのハイライトなど
        # ----------------------------------------------------------
        # 確定済み(visited)ノードをオレンジ、未確定を水色に
        node_colors = []
        for n in G.nodes:
            if n in visited:
                node_colors.append("orange")
            else:
                node_colors.append("lightblue")
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)

        # ノードラベル（都市名）
        nx.draw_networkx_labels(G, pos, ax=ax)

        # 暫定距離をノード下に表示
        for n in G.nodes:
            d = dist[n]
            if d < float('inf'):
                x, y = pos[n]
                ax.text(x, y - 0.1, f"{d:.0f}", size=9,
                        ha='center', va='center', color='red')

        # 現在確定したノードを赤色で強調
        if current is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[current], node_color='red', ax=ax)

        # 今回更新された隣接ノードを緑色で強調
        if neighbor is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[neighbor], node_color='green', ax=ax)

        # end_node が指定されていれば、最短経路を赤線で描画
        if end_node is not None and dist[end_node] < float('inf'):
            path_edges = reconstruct_path(prev, start_node, end_node)
            # 最短経路は単に red, rad=0.2 で
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2,
                                   ax=ax, arrows=True, arrowstyle='->', connectionstyle='arc3, rad=0.2')

        ax.set_xticks([])
        ax.set_yticks([])

    def update(frame):
        visited, dist, current, neighbor, prev = steps[frame]
        draw_graph(visited, dist, current, neighbor, prev)

    ani = FuncAnimation(fig, update, frames=range(len(steps)), interval=1000, repeat=False)
    ani.save('dijkstra_animation.gif', writer='pillow')
    print("アニメーションを 'dijkstra_animation.gif' として保存しました。")

# ----------------------------
# メイン実行部
# ----------------------------
def main():
    """
    CLIから始点と終点を選び、行きと帰りでコストが異なる有向グラフをダイクストラ法で可視化するサンプル
    """
    # ★行きと帰りが異なるコストを定義した隣接リスト (有向)
    places = {
        "Tokyo": {
            "Nagoya": 300,
            "Osaka": 500,
            "Sapporo": 800
        },
        "Nagoya": {
            "Tokyo": 350,
            "Osaka": 200,
            "Fukuoka": 600
        },
        "Osaka": {
            "Tokyo": 480,
            "Nagoya": 250,
            "Fukuoka": 600
        },
        "Fukuoka": {
            "Osaka": 620
        },
        "Sapporo": {
            "Tokyo": 880
            # Sapporo -> 他 は無いとする
        }
    }

    # 有向グラフを作成
    G, pos = create_graph_from_dict(places)

    # ノード一覧にIDを付与
    nodes = list(G.nodes)
    id_to_node = {i+1: n for i, n in enumerate(nodes)}
    node_to_id = {n: i+1 for i, n in enumerate(nodes)}

    # ノード一覧を表示
    print("ノード一覧:")
    for i, node in id_to_node.items():
        print(f"{i}: {node}")

    # CLIから始点IDと終点IDを入力
    while True:
        try:
            start_id = int(input("始点のIDを入力してください: "))
            if start_id not in id_to_node:
                raise ValueError
            break
        except ValueError:
            print("無効なIDです。再度入力してください。")

    while True:
        try:
            end_id = int(input("終点のIDを入力してください: "))
            if end_id not in id_to_node:
                raise ValueError
            break
        except ValueError:
            print("無効なIDです。再度入力してください。")

    start_node = id_to_node[start_id]
    end_node = id_to_node[end_id]

    print(f"始点: {start_node}, 終点: {end_node}")
    visualize_dijkstra(G, pos, start_node=start_node, end_node=end_node)

if __name__ == "__main__":
    main()
