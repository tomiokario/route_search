import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import sys

# ----------------------------
# グラフ構築（隣接リスト形式の辞書から）
# ----------------------------
def create_graph_from_dict(adj_dict):
    G = nx.Graph()
    for src, neighbors in adj_dict.items():
        G.add_node(src)
        for dst, w in neighbors.items():
            G.add_node(dst)
            G.add_edge(src, dst, weight=w)
    pos = nx.spring_layout(G, seed=42)
    return G, pos

# ----------------------------
# ダイクストラ法（ステップごとの状態を yield）
# ----------------------------
def dijkstra_steps(G, start, end=None):
    """
    ダイクストラ法を1ステップずつ進め、その都度の状態を yield する。

    戻り値 (各ステップ) のタプル:
        (
            visited,         # 確定したノード集合
            dist,            # 各ノードへの暫定距離
            current_node,    # 今回新しく確定したノード (隣接更新時は同じ値)
            neighbor_node,   # 距離を更新した隣接ノード (更新がある場合だけ実際のノード)
            prev,            # 前任ノードの辞書
        )
    """
    dist = {node: float('inf') for node in G.nodes}
    dist[start] = 0

    prev = {node: None for node in G.nodes}

    unvisited = set(G.nodes)
    visited = set()

    while unvisited:
        current_node = min(unvisited, key=lambda node: dist[node])
        if dist[current_node] == float('inf'):
            break

        unvisited.remove(current_node)
        visited.add(current_node)

        # "current_node" が確定した直後の状態を一旦 yield
        yield visited.copy(), dist.copy(), current_node, None, prev.copy()

        # 隣接ノードの距離を更新
        for neighbor in G[current_node]:
            if neighbor in unvisited:
                edge_weight = G[current_node][neighbor]['weight']
                new_distance = dist[current_node] + edge_weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    prev[neighbor] = current_node
                    # 更新があった場合、そのステータスを yield
                    yield visited.copy(), dist.copy(), current_node, neighbor, prev.copy()

    # 全ループ終了後に最終状態を一度 yield してもよい
    # yield visited.copy(), dist.copy(), None, None, prev.copy()

# ----------------------------
# 最短パス復元用関数
# ----------------------------
def reconstruct_path(prev, start, end):
    """
    `prev` をたどって start -> end の経路を (edge のリスト) として返す。
    例: [('Tokyo', 'Nagoya'), ('Nagoya', 'Fukuoka')] のような形。
    end から start に向けてさかのぼり、最後に逆順にして返す。
    """
    if start == end:
        return []
    path_edges = []
    node = end
    while node is not None and node != start:
        parent = prev[node]
        if parent is None:
            break
        path_edges.append((parent, node))
        node = parent

    # 逆順になっているので reverse
    path_edges.reverse()
    return path_edges

# ----------------------------
# 描画用 (アニメーション)
# ----------------------------
def visualize_dijkstra(G, pos, start_node, end_node=None, path_edges_final=None):
    """
    ダイクストラ法の進行をアニメーション表示。
    end_node を指定した場合、各ステップで end_node の最短経路が確定していれば赤色でハイライトします。
    """
    steps = list(dijkstra_steps(G, start_node, end_node))

    fig, ax = plt.subplots(figsize=(8, 6))

    def draw_graph(visited, dist, current, neighbor, prev):
        ax.clear()
        ax.set_title(f"Dijkstra's Algorithm (Start: {start_node}, End: {end_node})", fontsize=14)

        # 全エッジを薄いグレーで描画
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5)
        # 重み表示
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        # ノードの色分け: 確定済み (orange) / 未確定 (lightblue)
        node_colors = ['orange' if node in visited else 'lightblue' for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)

        # ノードラベル
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

        # 暫定距離を各ノードの下に表示
        for node in G.nodes:
            d = dist[node]
            if d < float('inf'):
                x, y = pos[node]
                ax.text(x, y - 0.1, f"{d:.0f}", size=9, ha='center', va='center', color='red')

        # 現在処理中のノードを赤色で強調
        if current is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[current], node_color='red', ax=ax)

        # 距離更新があった隣接ノードを緑色で強調
        if neighbor is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[neighbor], node_color='green', ax=ax)

        # 最終的な最短経路を赤色でハイライト
        if end_node is not None and path_edges_final is not None:
            nx.draw_networkx_edges(G, pos, edgelist=path_edges_final, edge_color='red', width=2, ax=ax)

        ax.set_xticks([])
        ax.set_yticks([])

    def update(frame):
        visited, dist, current, neighbor, prev = steps[frame]
        # 最終的なパスを復元
        if end_node is not None and prev[end_node] is not None:
            path_edges = reconstruct_path(prev, start_node, end_node)
        else:
            path_edges = []
        draw_graph(visited, dist, current, neighbor, prev)
        # 最終的なパスをハイライト
        if path_edges:
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)

    ani = FuncAnimation(fig, update, frames=range(len(steps)), interval=1000, repeat=False)
    # 対話的環境ならこちらで表示
    # plt.show()
    ani.save('dijkstra_animation.gif', writer='pillow')
    print("アニメーションを 'dijkstra_animation.gif' として保存しました。")

# ----------------------------
# メイン実行部
# ----------------------------
def main():
    # サンプル用データ
    places = {
        "Tokyo":   {"Nagoya": 350, "Osaka": 515},
        "Nagoya":  {"Tokyo": 350, "Osaka": 200, "Fukuoka": 550},
        "Osaka":   {"Tokyo": 515, "Nagoya": 200, "Fukuoka": 600},
        "Fukuoka": {"Nagoya": 550, "Osaka": 600, "Sapporo": 1150},
        "Sapporo": {"Fukuoka": 1150}
    }

    # グラフ作成
    G, pos = create_graph_from_dict(places)

    # ノードにIDを割り当て
    nodes = list(G.nodes)
    id_to_node = {i+1: node for i, node in enumerate(nodes)}
    node_to_id = {node: i+1 for i, node in enumerate(nodes)}

    # ノード一覧を表示
    print("ノード一覧:")
    for id, node in id_to_node.items():
        print(f"{id}: {node}")

    # ユーザーから始点と終点のIDを入力
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

    # 最短経路のアニメーションを作成
    visualize_dijkstra(G, pos, start_node=start_node, end_node=end_node)

if __name__ == "__main__":
    main()
