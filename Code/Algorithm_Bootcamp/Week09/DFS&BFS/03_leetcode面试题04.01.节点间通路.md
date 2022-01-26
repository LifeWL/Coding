#### [面试题 04.01. 节点间通路](https://leetcode-cn.com/problems/route-between-nodes-lcci/)

节点间通路。给定有向图，设计一个算法，找出两个节点之间是否存在一条路径。

**示例1:**

```
 输入：n = 3, graph = [[0, 1], [0, 2], [1, 2], [1, 2]], start = 0, target = 2
 输出：true
```

**示例2:**

```
 输入：n = 5, graph = [[0, 1], [0, 2], [0, 4], [0, 4], [0, 1], [1, 3], [1, 4], [1, 3], [2, 3], [3, 4]], start = 0, target = 4
 输出 true
```



```C++
class Solution {
public:
    vector<bool> visited;
    vector<unordered_set<int>> adj;
    bool found = false;
    bool findWhetherExistsPath(int n, vector<vector<int>>& graph, int start, int target) {
        visited.assign(n, false);
        adj.assign(n, unordered_set<int>());
        for (int i = 0; i < n; ++i) {
        auto it = adj[graph[i][0]].find(graph[i][1]);
        if (it == adj[graph[i][0]].end()) {
            adj[graph[i][0]].insert(graph[i][1]);
        }
        }
        dfs(start, target);
        return found;
    }
    void dfs(int cur, int target) {
        if (found) return;
        if (cur == target) {
            found = true;
            return;
        }
        visited[cur] = true;
        for (int next: adj[cur]) {
            if (!visited[next]) {
                dfs(next, target);
            }
        }
    }
};
```

