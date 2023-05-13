//快排
void quick_sort(int q[], int l, int r)
{
    if (l >= r) return;
    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j)
    {
        do i++; while (q[i] < x);
        do j--; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    quick_sort(q, l, j), quick_sort(q, j + 1, r);
}

//归并排序
void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;
    int mid = l + r >> 1;
    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);
    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
    {
        if (q[i] <= q[j]) tmp[k++] = q[i++];
        else tmp[k++] = q[j++];
    }
    while (i <= mid) tmp[k++] = q[i++];
    while (j <= r) tmp[k++] = q[j++];
    for (int i = l, j = 0; i <= r; i++, j++)
    {
        q[i] = tmp[j];
    }
}

//二分
bool check(double x){}

double bsearch_3(double l, double r)
{
    const double eps = 1e-6;
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}

//高精度加法
vector<int> add(vector<int> &A, vector<int> &B)
{ if (A.size() < B.size()) return add(B, A);
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i++)
    {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    if (t) C.push_back(t);
    return C;
}

//高精度减法
vector<int> sub(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i++)
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}

//高精度乘法
vector<int> mul(vector<int> &A, int b)
{
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size() || t; i++)
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}

//高精度除法
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i--)
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while(C.size() > 1 && C.back() == 0) c.pop_back();
    return C;
}

//一维前缀和
//S[i] = a[1] + a[2] + ... a[i]
//a[l] + ... + a[r] = S[r] - S[l - 1]

//二维前缀和
//S[i, j] = 第i行j列格子左上部分所有元素的和
//以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵的和为：
//S[x2, y2] - S[x1 - 1, y2] - S[x2, y1 - 1] + S[x1 - 1, y1 - 1]

//一维差分
//给区间[l, r]中的每个数加上c：B[l] += c, B[r + 1] -= c

//二维区分
//给以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵中的所有元素加上c：
//S[x1, y1] += c, S[x2 + 1, y1] -= c, S[x1, y2 + 1] -= c, S[x2 + 1, y2 + 1] += c

//位运算
//求n的第k位数字: n >> k & 1
//返回n的最后一位1：lowbit(n) = n & -n

//双指针
for (int i = 0, j = 0; i < n; i ++ )
{
    while (j < i && check(i, j)) j ++ ;

    // 具体问题的逻辑
}

//离散化
vector<int> alls; // 存储所有待离散化的值
sort(alls.begin(), alls.end()); // 将所有值排序
alls.erase(unique(alls.begin(), alls.end()), alls.end());   // 去掉重复元素

// 二分求出x对应的离散化的值
int find(int x) // 找到第一个大于等于x的位置
{
    int l = 0, r = alls.size() - 1;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1; // 映射到1, 2, ...n
}

//区间合并
void merge(vector<PII> &segs)
{
    vector<PII> res;

    sort(segs.begin(), segs.end());

    int st = -2e9, ed = -2e9;
    for (auto seg : segs)
        if (ed < seg.first)
        {
            if (st != -2e9) res.push_back({st, ed});
            st = seg.first, ed = seg.second;
        }
        else ed = max(ed, seg.second);

    if (st != -2e9) res.push_back({st, ed});

    segs = res;
}

//单链表
int head, e[N], ne[N], idx;

void init()
{
    head = -1;
    idx = 0;
}

//头插入
void insert(int a)
{
    e[idx] = a, ne[idx] = head, head = idx++;
}

//插入到k节点后
void insert(int k, int a)
{
    e[idx] = a, ne[idx] = ne[k], ne[k] = idx++;
}

void remove()
{
    head = ne[head];
}

void remove(int k)
{
    ne[k] = ne[ne[k]];
}

//双向链表

int e[N], l[N], r[N], idx;

void init()
{
    r[0] = 1, l[1] = 0;
}

void insert(int k, int x)
{
    e[idx] = x;
    l[idx] = k, r[idx] = r[k];
    l[r[k]] = idx, r[k] = idx++;
}

void remove(int k)
{
    l[r[k]] = l[k];
    r[l[k]] = r[k]; 
}


//栈
int stk[N], tt = 0;
//插入一个数
stk[++tt] = x;
//栈顶弹出
tt--;
//栈顶值
stk[tt];
//如果栈不为空
if (tt > 0) {}

//队列
int q[N], hh = 0, tt = -1;
q[++tt] = x;
//从对头弹出一个数
hh++;
//取对头的值
q[hh];
//队列不为空
if (hh <= tt) {}

//循环队列
int q[N], hh = 0, tt = 0;
//向队尾插入一个数
q[tt++] = x;
if (tt == N) tt = 0;
//弹出队头的数
hh++;
if (hh == N) hh = 0;
//队头的值
q[hh];
//队列不为空
if (hh != tt) {}

//单调栈
//常见模型：找出每个数左边离它最近的比它大/小的数
int tt = 0;
for (int i = 1; i <= n; i ++ )
{
    while (tt && check(stk[tt], i)) tt -- ;
    stk[ ++ tt] = i;
}

//单调队列
//常见模型：找出滑动窗口中的最大值/最小值
int hh = 0, tt = -1;
for (int i = 0; i < n; i ++ )
{
    while (hh <= tt && check_out(q[hh])) hh ++ ;  // 判断队头是否滑出窗口
    while (hh <= tt && check(q[tt], i)) tt -- ;
    q[ ++ tt] = i;
}

//KMP
//s[]是长文本，p[]是模式串，n是s的长度，m是p的长度
for (int i = 2, j = 0; i <= m; i++)
{
    while (j && p[i] != p[j + 1]) j = ne[j];
    if (p[i] == p[j + 1]) j++;
    ne[i] = j;
}

for (int i = 1, j = 0; i <= n; i++)
{
    while (j && s[i] != p[j + 1]) j = ne[j];
    if (s[i] == p[j + 1]) j++;
    if (j == m)
    {
        j = ne[j];
    }
}

//Trie树
int son[N][26], cnt[N], idx;

void insert(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++ ;
}

int query(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}


//普通并查集
int p[N];

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

for (int i = 1; i <= n; i++) p[i] = i;

p[find(a)] = find(b);


//维护size的并查集
int p[N], size[N];

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}


for (int i = 1; i <= n; i ++ )
{
    p[i] = i;
    size[i] = 1;
}


size[find(b)] += size[find(a)];
p[find(a)] = find(b);

//维护到祖宗节点距离的并查集
int p[N], d[N];

int find(int x)
{
    if (p[x] != x)
    {
        int u = find(p[x]);
        d[x] += d[p[x]];
        p[x] = u;
    }
    return p[x];
}

for (int i = 1; i <= n; ++i)
{
    p[i] = i;
    d[i] = 0;
}

p[find(a)] = find(b);
d[find(a)] = distance;

//堆
int h[N], ph[N], hp[N], size;

void heap_swap(int a, int b)
{
    swap(ph[hp[a]],ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}

void down(int u)
{
    int t = u;
    if (u * 2 <= size && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= size && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (u != t)
    {
        heap_swap(u, t);
        down(t);
    }
}

void up(int u)
{
    while (u / 2 && h[u] < h[u / 2])
    {
        heap_swap(u, u / 2);
        u >>= 1;
    }
}

for (int i = n / 2; i; i -- ) down(i);

//拉链哈希
int h[N], e[N], ne[N], idx;

void insert(int x)
{
    int k = (x % N + N) % N;
    e[idx] = x;
    ne[idx] = h[k];
    h[k] = idx++;
}

bool find(int x)
{
    int k = (x % N + N) % N;
    for (int i = h[k]; i != -1; i = ne[i])
    {
        if (e[i] == x)
        {
            return true;
        }
    }
    return false;
}

//开放寻址哈希
int h[N];

int find(int x)
{
    int t = (x % N + N) % N;
    while (h[t] != null && h[t] != x)
    {
        t++;
        if (t == N) t = 0;
    }
    return t;
}

//字典哈希
typedef unsigned long long ULL;
ULL h[N], p[N];
P = 131;
p[0] = 1;
for (int i = 1; i <= n; ++i)
{
    h[i] = h[i - 1] * P + str[i];
    p[i] = p[i - 1] * P;
}

ULL get(int l, int r)
{
    return h[r] - h[l - 1] * p[r - l + 1];
}

//树与图的存储
//树是一种特殊的图，与图的存储方式相同。
//对于无向图中的边ab，存储两条有向边a->b, b->a。
//因此我们可以只考虑有向图的存储。

//(1) 邻接矩阵：g[a][b] 存储边a->b

//(2) 邻接表：

// 对于每个点k，开一个单链表，存储k所有可以走到的点。h[k]存储这个单链表的头结点
int h[N], e[N], ne[N], idx;

// 添加一条边a->b
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 初始化
idx = 0;
memset(h, -1, sizeof h);

//深度优先遍历

int dfs(int u)
{
    st[u] = true; // st[u] 表示点u已经被遍历过

    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j]) dfs(j);
    }
}


//宽度优先遍历
queue<int> q;
st[1] = true; // 表示1号点已经被遍历过
q.push(1);

while (q.size())
{
    int t = q.front();
    q.pop();

    for (int i = h[t]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j])
        {
            st[j] = true; // 表示点j已经被遍历过
            q.push(j);
        }
    }
}

//拓扑排序
//时间复杂度 O(n+m)O(n+m), nn 表示点数，mm 表示边数
bool topsort()
{
    int hh = 0, tt = -1;

    // d[i] 存储点i的入度
    for (int i = 1; i <= n; i ++ )
        if (!d[i])
            q[ ++ tt] = i;

    while (hh <= tt)
    {
        int t = q[hh ++ ];

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (-- d[j] == 0)
                q[ ++ tt] = j;
        }
    }

    // 如果所有点都入队了，说明存在拓扑序列；否则不存在拓扑序列。
    return tt == n - 1;
}

//朴素dijkstra算法
//时间复杂是 O(n2+m)O(n2+m), nn 表示点数，mm 表示边数
int g[N][N];  // 存储每条边
int dist[N];  // 存储1号点到每个点的最短距离
bool st[N];   // 存储每个点的最短路是否已经确定

// 求1号点到n号点的最短路，如果不存在则返回-1
int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    for (int i = 0; i < n - 1; i ++ )
    {
        int t = -1;     // 在还未确定最短路的点中，寻找距离最小的点
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;

        // 用t更新其他点的距离
        for (int j = 1; j <= n; j ++ )
            dist[j] = min(dist[j], dist[t] + g[t][j]);

        st[t] = true;
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

//堆优化版dijkstra
//时间复杂度 O(mlogn)O(mlogn), nn 表示点数，mm 表示边数
typedef pair<int, int> PII;

int n;      // 点的数量
int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
int dist[N];        // 存储所有点到1号点的距离
bool st[N];     // 存储每个点的最短距离是否已确定

// 求1号点到n号点的最短距离，如果不存在，则返回-1
int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});      // first存储距离，second存储节点编号

    while (heap.size())
    {
        auto t = heap.top();
        heap.pop();

        int ver = t.second, distance = t.first;

        if (st[ver]) continue;
        st[ver] = true;

        for (int i = h[ver]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > distance + w[i])
            {
                dist[j] = distance + w[i];
                heap.push({dist[j], j});
            }
        }
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

//Bellman-Ford算法
//时间复杂度 O(nm)O(nm), nn 表示点数，mm 表示边数

int n, m;       // n表示点数，m表示边数
int dist[N];        // dist[x]存储1到x的最短路距离

struct Edge     // 边，a表示出点，b表示入点，w表示边的权重
{
    int a, b, w;
}edges[M];

// 求1到n的最短路距离，如果无法从1走到n，则返回-1。
int bellman_ford()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    // 如果第n次迭代仍然会松弛三角不等式，就说明存在一条长度是n+1的最短路径，由抽屉原理，路径中至少存在两个相同的点，说明图中存在负权回路。
    for (int i = 0; i < n; i ++ )
    {
        for (int j = 0; j < m; j ++ )
        {
            int a = edges[j].a, b = edges[j].b, w = edges[j].w;
            if (dist[b] > dist[a] + w)
                dist[b] = dist[a] + w;
        }
    }

    if (dist[n] > 0x3f3f3f3f / 2) return -1;
    return dist[n];
}

//spfa 算法（队列优化的Bellman-Ford算法
//时间复杂度 平均情况下 O(m)O(m)，最坏情况下 O(nm)O(nm), nn 表示点数，mm 表示边数
int n;      // 总点数
int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
int dist[N];        // 存储每个点到1号点的最短距离
bool st[N];     // 存储每个点是否在队列中

// 求1号点到n号点的最短路距离，如果从1号点无法走到n号点则返回-1
int spfa()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    queue<int> q;
    q.push(1);
    st[1] = true;

    while (q.size())
    {
        auto t = q.front();
        q.pop();

        st[t] = false;

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                if (!st[j])     // 如果队列中已存在j，则不需要将j重复插入
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

//spfa判断图中是否存在负环
//时间复杂度是 O(nm)O(nm), nn 表示点数，mm 表示边数
int n;      // 总点数
int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
int dist[N], cnt[N];        // dist[x]存储1号点到x的最短距离，cnt[x]存储1到x的最短路中经过的点数
bool st[N];     // 存储每个点是否在队列中

// 如果存在负环，则返回true，否则返回false。
bool spfa()
{
    // 不需要初始化dist数组
    // 原理：如果某条最短路径上有n个点（除了自己），那么加上自己之后一共有n+1个点，由抽屉原理一定有两个点相同，所以存在环。

    queue<int> q;
    for (int i = 1; i <= n; i ++ )
    {
        q.push(i);
        st[i] = true;
    }

    while (q.size())
    {
        auto t = q.front();
        q.pop();

        st[t] = false;

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;
                if (cnt[j] >= n) return true;       // 如果从1号点到x的最短路中包含至少n个点（不包括自己），则说明存在环
                if (!st[j])
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }

    return false;
}

//floyd算法
//时间复杂度是 O(n3)O(n3), nn 表示点数
//初始化：
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            if (i == j) d[i][j] = 0;
            else d[i][j] = INF;

// 算法结束后，d[a][b]表示a到b的最短距离
void floyd()
{
    for (int k = 1; k <= n; k ++ )
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
}

//朴素版prim算法
//时间复杂度是 O(n2+m)O(n2+m), nn 表示点数，mm 表示边数
int n;      // n表示点数
int g[N][N];        // 邻接矩阵，存储所有边
int dist[N];        // 存储其他点到当前最小生成树的距离
bool st[N];     // 存储每个点是否已经在生成树中


// 如果图不连通，则返回INF(值是0x3f3f3f3f), 否则返回最小生成树的树边权重之和
int prim()
{
    memset(dist, 0x3f, sizeof dist);

    int res = 0;
    for (int i = 0; i < n; i ++ )
    {
        int t = -1;
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;

        if (i && dist[t] == INF) return INF;

        if (i) res += dist[t];
        st[t] = true;

        for (int j = 1; j <= n; j ++ ) dist[j] = min(dist[j], g[t][j]);
    }

    return res;
}

//Kruskal算法
//时间复杂度是 O(mlogm)O(mlogm), nn 表示点数，mm 表示边数
int n, m;       // n是点数，m是边数
int p[N];       // 并查集的父节点数组

struct Edge     // 存储边
{
    int a, b, w;

    bool operator< (const Edge &W)const
    {
        return w < W.w;
    }
}edges[M];

int find(int x)     // 并查集核心操作
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int kruskal()
{
    sort(edges, edges + m);

    for (int i = 1; i <= n; i ++ ) p[i] = i;    // 初始化并查集

    int res = 0, cnt = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;

        a = find(a), b = find(b);
        if (a != b)     // 如果两个连通块不连通，则将这两个连通块合并
        {
            p[a] = b;
            res += w;
            cnt ++ ;
        }
    }

    if (cnt < n - 1) return INF;
    return res;
}

//染色法判别二分图
//时间复杂度是 O(n+m)O(n+m), nn 表示点数，mm 表示边数
int n;      // n表示点数
int h[N], e[M], ne[M], idx;     // 邻接表存储图
int color[N];       // 表示每个点的颜色，-1表示未染色，0表示白色，1表示黑色

// 参数：u表示当前节点，c表示当前点的颜色
bool dfs(int u, int c)
{
    color[u] = c;
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (color[j] == -1)
        {
            if (!dfs(j, !c)) return false;
        }
        else if (color[j] == c) return false;
    }

    return true;
}

bool check()
{
    memset(color, -1, sizeof color);
    bool flag = true;
    for (int i = 1; i <= n; i ++ )
        if (color[i] == -1)
            if (!dfs(i, 0))
            {
                flag = false;
                break;
            }
    return flag;
}

//匈牙利算法
//时间复杂度是 O(nm)O(nm), nn 表示点数，mm 表示边数
int n1, n2;     // n1表示第一个集合中的点数，n2表示第二个集合中的点数
int h[N], e[M], ne[M], idx;     // 邻接表存储所有边，匈牙利算法中只会用到从第一个集合指向第二个集合的边，所以这里只用存一个方向的边
int match[N];       // 存储第二个集合中的每个点当前匹配的第一个集合中的点是哪个
bool st[N];     // 表示第二个集合中的每个点是否已经被遍历过

bool find(int x)
{
    for (int i = h[x]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j])
        {
            st[j] = true;
            if (match[j] == 0 || find(match[j]))
            {
                match[j] = x;
                return true;
            }
        }
    }

    return false;
}

// 求最大匹配数，依次枚举第一个集合中的每个点能否匹配第二个集合中的点
int res = 0;
for (int i = 1; i <= n1; i ++ )
{
    memset(st, false, sizeof st);
    if (find(i)) res ++ ;
}

//试除法判定质数
bool is_prime(int x)
{
    if (x < 2) return false;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
            return false;
    return true;
}

//试除法分解质因数
void divide(int x)
{
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            int s = 0;
            while (x % i == 0) x /= i, s ++ ;
            cout << i << ' ' << s << endl;
        }
    if (x > 1) cout << x << ' ' << 1 << endl;
    cout << endl;
}

//朴素筛法求素数
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (st[i]) continue;
        primes[cnt ++ ] = i;
        for (int j = i + i; j <= n; j += i)
            st[j] = true;
    }
}

//线性筛法求素数
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}

//试除法求所有约数
vector<int> get_divisors(int x)
{
    vector<int> res;
    for (int i = 1; i <= x / i; i ++ )
        if (x % i == 0)
        {
            res.push_back(i);
            if (i != x / i) res.push_back(x / i);
        }
    sort(res.begin(), res.end());
    return res;
}

//约数个数和约数之和
//如果 N = p1^c1 * p2^c2 * ... *pk^ck
//约数个数： (c1 + 1) * (c2 + 1) * ... * (ck + 1)
//约数之和： (p1^0 + p1^1 + ... + p1^c1) * ... * (pk^0 + pk^1 + ... + pk^ck)

//欧几里得算法
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}

//求欧拉函数
int phi(int x)
{
    int res = x;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            res = res / i * (i - 1);
            while (x % i == 0) x /= i;
        }
    if (x > 1) res = res / x * (x - 1);

    return res;
}

//筛法求欧拉函数
int primes[N], cnt;     // primes[]存储所有素数
int euler[N];           // 存储每个数的欧拉函数
bool st[N];         // st[x]存储x是否被筛掉


void get_eulers(int n)
{
    euler[1] = 1;
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i])
        {
            primes[cnt ++ ] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            int t = primes[j] * i;
            st[t] = true;
            if (i % primes[j] == 0)
            {
                euler[t] = euler[i] * primes[j];
                break;
            }
            euler[t] = euler[i] * (primes[j] - 1);
        }
    }
}

//快速幂
//求 m^k mod p，时间复杂度 O(logk)。

int qmi(int m, int k, int p)
{
    int res = 1 % p, t = m;
    while (k)
    {
        if (k&1) res = res * t % p;
        t = t * t % p;
        k >>= 1;
    }
    return res;
}

//扩展欧几里得算法
// 求x, y，使得ax + by = gcd(a, b)
int exgcd(int a, int b, int &x, int &y)
{
    if (!b)
    {
        x = 1; y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= (a/b) * x;
    return d;
}

//高斯消元
// a[N][N]是增广矩阵
int gauss()
{
    int c, r;
    for (c = 0, r = 0; c < n; c ++ )
    {
        int t = r;
        for (int i = r; i < n; i ++ )   // 找到绝对值最大的行
            if (fabs(a[i][c]) > fabs(a[t][c]))
                t = i;

        if (fabs(a[t][c]) < eps) continue;

        for (int i = c; i <= n; i ++ ) swap(a[t][i], a[r][i]);      // 将绝对值最大的行换到最顶端
        for (int i = n; i >= c; i -- ) a[r][i] /= a[r][c];      // 将当前行的首位变成1
        for (int i = r + 1; i < n; i ++ )       // 用当前行将下面所有的列消成0
            if (fabs(a[i][c]) > eps)
                for (int j = n; j >= c; j -- )
                    a[i][j] -= a[r][j] * a[i][c];

        r ++ ;
    }

    if (r < n)
    {
        for (int i = r; i < n; i ++ )
            if (fabs(a[i][n]) > eps)
                return 2; // 无解
        return 1; // 有无穷多组解
    }

    for (int i = n - 1; i >= 0; i -- )
        for (int j = i + 1; j < n; j ++ )
            a[i][n] -= a[i][j] * a[j][n];

    return 0; // 有唯一解
}

//递推法求组合数
// c[a][b] 表示从a个苹果中选b个的方案数
for (int i = 0; i < N; i ++ )
    for (int j = 0; j <= i; j ++ )
        if (!j) c[i][j] = 1;
        else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;


//通过预处理逆元的方式求组合数
//首先预处理出所有阶乘取模的余数fact[N]，以及所有阶乘取模的逆元infact[N]
//如果取模的数是质数，可以用费马小定理求逆元
int qmi(int a, int k, int p)    // 快速幂模板
{
    int res = 1;
    while (k)
    {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

// 预处理阶乘的余数和阶乘逆元的余数
fact[0] = infact[0] = 1;
for (int i = 1; i < N; i ++ )
{
    fact[i] = (LL)fact[i - 1] * i % mod;
    infact[i] = (LL)infact[i - 1] * qmi(i, mod - 2, mod) % mod;
}


//Lucas定理
//若p是质数，则对于任意整数 1 <= m <= n，有：
    C(n, m) = C(n % p, m % p) * C(n / p, m / p) (mod p)

int qmi(int a, int k, int p)  // 快速幂模板
{
    int res = 1 % p;
    while (k)
    {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

int C(int a, int b, int p)  // 通过定理求组合数C(a, b)
{
    if (a < b) return 0;

    LL x = 1, y = 1;  // x是分子，y是分母
    for (int i = a, j = 1; j <= b; i --, j ++ )
    {
        x = (LL)x * i % p;
        y = (LL) y * j % p;
    }

    return x * (LL)qmi(y, p - 2, p) % p;
}

int lucas(LL a, LL b, int p)
{
    if (a < p && b < p) return C(a, b, p);
    return (LL)C(a % p, b % p, p) * lucas(a / p, b / p, p) % p;
}


//分解质因数法求组合数
//当我们需要求出组合数的真实值，而非对某个数的余数时，分解质因数的方式比较好用：
//   1. 筛法求出范围内的所有质数
//   2. 通过 C(a, b) = a! / b! / (a - b)! 这个公式求出每个质因子的次数。 n! 中p的次数是 n / p + n / p^2 + n / p^3 + ...
//   3. 用高精度乘法将所有质因子相乘

int primes[N], cnt;     // 存储所有质数
int sum[N];     // 存储每个质数的次数
bool st[N];     // 存储每个数是否已被筛掉


void get_primes(int n)      // 线性筛法求素数
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}


int get(int n, int p)       // 求n！中的次数
{
    int res = 0;
    while (n)
    {
        res += n / p;
        n /= p;
    }
    return res;
}


vector<int> mul(vector<int> a, int b)       // 高精度乘低精度模板
{
    vector<int> c;
    int t = 0;
    for (int i = 0; i < a.size(); i ++ )
    {
        t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }

    while (t)
    {
        c.push_back(t % 10);
        t /= 10;
    }

    return c;
}

get_primes(a);  // 预处理范围内的所有质数

for (int i = 0; i < cnt; i ++ )     // 求每个质因数的次数
{
    int p = primes[i];
    sum[i] = get(a, p) - get(b, p) - get(a - b, p);
}

vector<int> res;
res.push_back(1);

for (int i = 0; i < cnt; i ++ )     // 用高精度乘法将所有质因子相乘
    for (int j = 0; j < sum[i]; j ++ )
        res = mul(res, primes[i]);


//卡特兰数
//给定n个0和n个1，它们按照某种顺序排成长度为2n的序列，满足任意前缀中0的个数都不少于1的个数的序列的数量为： Cat(n) = C(2n, n) / (n + 1)
//NIM游戏
//给定N堆物品，第i堆物品有Ai个。两名玩家轮流行动，每次可以任选一堆，取走任意多个物品，可把一堆取光，但不能不取。取走最后一件物品者获胜。两人都采取最优策略，问先手是否必胜。
//我们把这种游戏称为NIM博弈。把游戏过程中面临的状态称为局面。整局游戏第一个行动的称为先手，第二个行动的称为后手。若在某一局面下无论采取何种行动，都会输掉游戏，则称该局面必败。
//所谓采取最优策略是指，若在某一局面下存在某种行动，使得行动后对面面临必败局面，则优先采取该行动。同时，这样的局面被称为必胜。我们讨论的博弈问题一般都只考虑理想情况，即两人均无失误，都采取最优策略行动时游戏的结果。
//NIM博弈不存在平局，只有先手必胜和先手必败两种情况。
//定理： NIM博弈先手必胜，当且仅当 A1 ^ A2 ^ … ^ An != 0


//树状数组
int n;
int a[N];
int tr[N];
int Greater[N], lower[N];

int lowbit(int x)
{
    return x & -x;
}

void add(int x, int c)
{
    for (int i = x; i <= n; i += lowbit(i)) tr[i] += c;
}

int sum(int x)
{
    int res = 0;
    for (int i = x; i; i -= lowbit(i)) res += tr[i];
    return res;
}

int main()
{
    scanf("%d", &n);

    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);

    for (int i = 1; i <= n; i ++ )
    {
        int y = a[i];
        Greater[i] = sum(n) - sum(y);
        lower[i] = sum(y - 1);
        add(y, 1);
    }

    memset(tr, 0, sizeof tr);
    LL res1 = 0, res2 = 0;
    for (int i = n; i; i -- )
    {
        int y = a[i];
        res1 += Greater[i] * (LL)(sum(n) - sum(y));
        res2 += lower[i] * (LL)(sum(y - 1));
        add(y, 1);
    }

    printf("%lld %lld\n", res1, res2);

    return 0;
}


//线段树
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 200010;

int m, p;
struct Node
{
    int l, r;
    int v;  // 区间[l, r]中的最大值
}tr[N * 4];

void pushup(int u)  // 由子节点的信息，来计算父节点的信息
{
    tr[u].v = max(tr[u << 1].v, tr[u << 1 | 1].v);
}

void build(int u, int l, int r)
{
    tr[u] = {l, r};
    if (l == r) return;
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
}

int query(int u, int l, int r)
{
    if (tr[u].l >= l && tr[u].r <= r) return tr[u].v;   // 树中节点，已经被完全包含在[l, r]中了

    int mid = tr[u].l + tr[u].r >> 1;
    int v = 0;
    if (l <= mid) v = query(u << 1, l, r);
    if (r > mid) v = max(v, query(u << 1 | 1, l, r));

    return v;
}

void modify(int u, int x, int v)
{
    if (tr[u].l == x && tr[u].r == x) tr[u].v = v;
    else
    {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid) modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}


int main()
{
    int n = 0, last = 0;
    scanf("%d%d", &m, &p);
    build(1, 1, m);

    int x;
    char op[2];
    while (m -- )
    {
        scanf("%s%d", op, &x);
        if (*op == 'Q')
        {
            last = query(1, n - x + 1, n);
            printf("%d\n", last);
        }
        else
        {
            modify(1, n + 1, ((LL)last + x) % p);
            n ++ ;
        }
    }

    return 0;
}

//可持久化数据结构
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 600010, M = N * 25;

int n, m;
int s[N];
int tr[M][2], max_id[M];
int root[N], idx;

void insert(int i, int k, int p, int q)
{
    if (k < 0)
    {
        max_id[q] = i;
        return;
    }
    int v = s[i] >> k & 1;
    if (p) tr[q][v ^ 1] = tr[p][v ^ 1];
    tr[q][v] = ++ idx;
    insert(i, k - 1, tr[p][v], tr[q][v]);
    max_id[q] = max(max_id[tr[q][0]], max_id[tr[q][1]]);
}

int query(int root, int C, int L)
{
    int p = root;
    for (int i = 23; i >= 0; i -- )
    {
        int v = C >> i & 1;
        if (max_id[tr[p][v ^ 1]] >= L) p = tr[p][v ^ 1];
        else p = tr[p][v];
    }

    return C ^ s[max_id[p]];
}

int main()
{
    scanf("%d%d", &n, &m);

    max_id[0] = -1;
    root[0] = ++ idx;
    insert(0, 23, 0, root[0]);

    for (int i = 1; i <= n; i ++ )
    {
        int x;
        scanf("%d", &x);
        s[i] = s[i - 1] ^ x;
        root[i] = ++ idx;
        insert(i, 23, root[i - 1], root[i]);
    }

    char op[2];
    int l, r, x;
    while (m -- )
    {
        scanf("%s", op);
        if (*op == 'A')
        {
            scanf("%d", &x);
            n ++ ;
            s[n] = s[n - 1] ^ x;
            root[n] = ++ idx;
            insert(n, 23, root[n - 1], root[n]);
        }
        else
        {
            scanf("%d%d%d", &l, &r, &x);
            printf("%d\n", query(root[r - 1], s[n] ^ x, l - 1));
        }
    }

    return 0;
}

//平衡树
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010, INF = 1e8;

int n;
struct Node
{
    int l, r;
    int key, val;
    int cnt, size;
}tr[N];

int root, idx;

void pushup(int p)
{
    tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + tr[p].cnt;
}

int get_node(int key)
{
    tr[ ++ idx].key = key;
    tr[idx].val = rand();
    tr[idx].cnt = tr[idx].size = 1;
    return idx;
}

void zig(int &p)    // 右旋
{
    int q = tr[p].l;
    tr[p].l = tr[q].r, tr[q].r = p, p = q;
    pushup(tr[p].r), pushup(p);
}

void zag(int &p)    // 左旋
{
    int q = tr[p].r;
    tr[p].r = tr[q].l, tr[q].l = p, p = q;
    pushup(tr[p].l), pushup(p);
}

void build()
{
    get_node(-INF), get_node(INF);
    root = 1, tr[1].r = 2;
    pushup(root);

    if (tr[1].val < tr[2].val) zag(root);
}


void insert(int &p, int key)
{
    if (!p) p = get_node(key);
    else if (tr[p].key == key) tr[p].cnt ++ ;
    else if (tr[p].key > key)
    {
        insert(tr[p].l, key);
        if (tr[tr[p].l].val > tr[p].val) zig(p);
    }
    else
    {
        insert(tr[p].r, key);
        if (tr[tr[p].r].val > tr[p].val) zag(p);
    }
    pushup(p);
}

void remove(int &p, int key)
{
    if (!p) return;
    if (tr[p].key == key)
    {
        if (tr[p].cnt > 1) tr[p].cnt -- ;
        else if (tr[p].l || tr[p].r)
        {
            if (!tr[p].r || tr[tr[p].l].val > tr[tr[p].r].val)
            {
                zig(p);
                remove(tr[p].r, key);
            }
            else
            {
                zag(p);
                remove(tr[p].l, key);
            }
        }
        else p = 0;
    }
    else if (tr[p].key > key) remove(tr[p].l, key);
    else remove(tr[p].r, key);

    pushup(p);
}

int get_rank_by_key(int p, int key)    // 通过数值找排名
{
    if (!p) return 0;   // 本题中不会发生此情况
    if (tr[p].key == key) return tr[tr[p].l].size + 1;
    if (tr[p].key > key) return get_rank_by_key(tr[p].l, key);
    return tr[tr[p].l].size + tr[p].cnt + get_rank_by_key(tr[p].r, key);
}

int get_key_by_rank(int p, int rank)   // 通过排名找数值
{
    if (!p) return INF;     // 本题中不会发生此情况
    if (tr[tr[p].l].size >= rank) return get_key_by_rank(tr[p].l, rank);
    if (tr[tr[p].l].size + tr[p].cnt >= rank) return tr[p].key;
    return get_key_by_rank(tr[p].r, rank - tr[tr[p].l].size - tr[p].cnt);
}

int get_prev(int p, int key)   // 找到严格小于key的最大数
{
    if (!p) return -INF;
    if (tr[p].key >= key) return get_prev(tr[p].l, key);
    return max(tr[p].key, get_prev(tr[p].r, key));
}

int get_next(int p, int key)    // 找到严格大于key的最小数
{
    if (!p) return INF;
    if (tr[p].key <= key) return get_next(tr[p].r, key);
    return min(tr[p].key, get_next(tr[p].l, key));
}

int main()
{
    build();

    scanf("%d", &n);
    while (n -- )
    {
        int opt, x;
        scanf("%d%d", &opt, &x);
        if (opt == 1) insert(root, x);
        else if (opt == 2) remove(root, x);
        else if (opt == 3) printf("%d\n", get_rank_by_key(root, x) - 1);
        else if (opt == 4) printf("%d\n", get_key_by_rank(root, x + 1));
        else if (opt == 5) printf("%d\n", get_prev(root, x));
        else printf("%d\n", get_next(root, x));
    }

    return 0;
}

//AC自动机
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 10010, S = 55, M = 1000010;

int n;
int tr[N * S][26], cnt[N * S], idx;
char str[M];
int q[N * S], ne[N * S];

void insert()
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int t = str[i] - 'a';
        if (!tr[p][t]) tr[p][t] = ++ idx;
        p = tr[p][t];
    }
    cnt[p] ++ ;
}

void build()
{
    int hh = 0, tt = -1;
    for (int i = 0; i < 26; i ++ )
        if (tr[0][i])
            q[ ++ tt] = tr[0][i];

    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = 0; i < 26; i ++ )
        {
            int p = tr[t][i];
            if (!p) tr[t][i] = tr[ne[t]][i];
            else
            {
                ne[p] = tr[ne[t]][i];
                q[ ++ tt] = p;
            }
        }
    }
}

int main()
{
    int T;
    scanf("%d", &T);
    while (T -- )
    {
        memset(tr, 0, sizeof tr);
        memset(cnt, 0, sizeof cnt);
        memset(ne, 0, sizeof ne);
        idx = 0;

        scanf("%d", &n);
        for (int i = 0; i < n; i ++ )
        {
            scanf("%s", str);
            insert();
        }

        build();

        scanf("%s", str);

        int res = 0;
        for (int i = 0, j = 0; str[i]; i ++ )
        {
            int t = str[i] - 'a';
            j = tr[j][t];

            int p = j;
            while (p)
            {
                res += cnt[p];
                cnt[p] = 0;
                p = ne[p];
            }
        }

        printf("%d\n", res);
    }

    return 0;
}

//splay
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n, m;
struct Node
{
    int s[2], p, v;
    int size, flag;

    void init(int _v, int _p)
    {
        v = _v, p = _p;
        size = 1;
    }
}tr[N];
int root, idx;

void pushup(int x)
{
    tr[x].size = tr[tr[x].s[0]].size + tr[tr[x].s[1]].size + 1;
}

void pushdown(int x)
{
    if (tr[x].flag)
    {
        swap(tr[x].s[0], tr[x].s[1]);
        tr[tr[x].s[0]].flag ^= 1;
        tr[tr[x].s[1]].flag ^= 1;
        tr[x].flag = 0;
    }
}

void rotate(int x)
{
    int y = tr[x].p, z = tr[y].p;
    int k = tr[y].s[1] == x;  // k=0表示x是y的左儿子；k=1表示x是y的右儿子
    tr[z].s[tr[z].s[1] == y] = x, tr[x].p = z;
    tr[y].s[k] = tr[x].s[k ^ 1], tr[tr[x].s[k ^ 1]].p = y;
    tr[x].s[k ^ 1] = y, tr[y].p = x;
    pushup(y), pushup(x);
}

void splay(int x, int k)
{
    while (tr[x].p != k)
    {
        int y = tr[x].p, z = tr[y].p;
        if (z != k)
            if ((tr[y].s[1] == x) ^ (tr[z].s[1] == y)) rotate(x);
            else rotate(y);
        rotate(x);
    }
    if (!k) root = x;
}

void insert(int v)
{
    int u = root, p = 0;
    while (u) p = u, u = tr[u].s[v > tr[u].v];
    u = ++ idx;
    if (p) tr[p].s[v > tr[p].v] = u;
    tr[u].init(v, p);
    splay(u, 0);
}

int get_k(int k)
{
    int u = root;
    while (true)
    {
        pushdown(u);
        if (tr[tr[u].s[0]].size >= k) u = tr[u].s[0];
        else if (tr[tr[u].s[0]].size + 1 == k) return u;
        else k -= tr[tr[u].s[0]].size + 1, u = tr[u].s[1];
    }
    return -1;
}

void output(int u)
{
    pushdown(u);
    if (tr[u].s[0]) output(tr[u].s[0]);
    if (tr[u].v >= 1 && tr[u].v <= n) printf("%d ", tr[u].v);
    if (tr[u].s[1]) output(tr[u].s[1]);
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 0; i <= n + 1; i ++ ) insert(i);
    while (m -- )
    {
        int l, r;
        scanf("%d%d", &l, &r);
        l = get_k(l), r = get_k(r + 2);
        splay(l, 0), splay(r, l);
        tr[tr[r].s[0]].flag ^= 1;
    }
    output(root);
    return 0;
}

#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010, INF = 1e9;

int n, m, delta;
struct Node
{
    int s[2], p, v;
    int size;

    void init(int _v, int _p)
    {
        v = _v, p = _p;
        size = 1;
    }
}tr[N];
int root, idx;

void pushup(int x)
{
    tr[x].size = tr[tr[x].s[0]].size + tr[tr[x].s[1]].size + 1;
}

void rotate(int x)
{
    int y = tr[x].p, z = tr[y].p;
    int k = tr[y].s[1] == x;
    tr[z].s[tr[z].s[1] == y] = x, tr[x].p = z;
    tr[y].s[k] = tr[x].s[k ^ 1], tr[tr[x].s[k ^ 1]].p = y;
    tr[x].s[k ^ 1] = y, tr[y].p = x;
    pushup(y), pushup(x);
}

void splay(int x, int k)
{
    while (tr[x].p != k)
    {
        int y = tr[x].p, z = tr[y].p;
        if (z != k)
            if ((tr[y].s[1] == x) ^ (tr[z].s[1] == y)) rotate(x);
            else rotate(y);
        rotate(x);
    }
    if (!k) root = x;
}

int insert(int v)
{
    int u = root, p = 0;
    while (u) p = u, u = tr[u].s[v > tr[u].v];
    u = ++ idx;
    if (p) tr[p].s[v > tr[p].v] = u;
    tr[u].init(v, p);
    splay(u, 0);
    return u;
}

int get(int v)
{
    int u = root, res;
    while (u)
    {
        if (tr[u].v >= v) res = u, u = tr[u].s[0];
        else u = tr[u].s[1];
    }
    return res;
}

int get_k(int k)
{
    int u = root;
    while (u)
    {
        if (tr[tr[u].s[0]].size >= k) u = tr[u].s[0];
        else if (tr[tr[u].s[0]].size + 1 == k) return tr[u].v;
        else k -= tr[tr[u].s[0]].size + 1, u = tr[u].s[1];
    }
    return -1;
}

int main()
{
    scanf("%d%d", &n, &m);
    int L = insert(-INF), R = insert(INF);

    int tot = 0;
    while (n -- )
    {
        char op[2];
        int k;
        scanf("%s%d", op, &k);
        if (*op == 'I')
        {
            if (k >= m) k -= delta, insert(k), tot ++ ;
        }
        else if (*op == 'A') delta += k;
        else if (*op == 'S')
        {
            delta -= k;
            R = get(m - delta);
            splay(R, 0), splay(L, R);
            tr[L].s[1] = 0;
            pushup(L), pushup(R);
        }
        else
        {
            if (tr[root].size - 2 < k) puts("-1");
            else printf("%d\n", get_k(tr[root].size - k) + delta);
        }
    }

    printf("%d\n", tot - (tr[root].size - 2));

    return 0;
}

#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1800010;

int n, m;
struct Node
{
    int s[2], p, v, id;
    int size;

    void init(int _v, int _id, int _p)
    {
        v = _v, id = _id, p = _p;
        size = 1;
    }
}tr[N];
int root[N], idx;
int p[N];

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

void pushup(int x)
{
    tr[x].size = tr[tr[x].s[0]].size + tr[tr[x].s[1]].size + 1;
}

void rotate(int x)
{
    int y = tr[x].p, z = tr[y].p;
    int k = tr[y].s[1] == x;
    tr[z].s[tr[z].s[1] == y] = x, tr[x].p = z;
    tr[y].s[k] = tr[x].s[k ^ 1], tr[tr[x].s[k ^ 1]].p = y;
    tr[x].s[k ^ 1] = y, tr[y].p = x;
    pushup(y), pushup(x);
}

void splay(int x, int k, int b)
{
    while (tr[x].p != k)
    {
        int y = tr[x].p, z = tr[y].p;
        if (z != k)
            if ((tr[y].s[1] == x) ^ (tr[z].s[1] == y)) rotate(x);
            else rotate(y);
        rotate(x);
    }
    if (!k) root[b] = x;
}

void insert(int v, int id, int b)
{
    int u = root[b], p = 0;
    while (u) p = u, u = tr[u].s[v > tr[u].v];
    u = ++ idx;
    if (p) tr[p].s[v > tr[p].v] = u;
    tr[u].init(v, id, p);
    splay(u, 0, b);
}

int get_k(int k, int b)
{
    int u = root[b];
    while (u)
    {
        if (tr[tr[u].s[0]].size >= k) u = tr[u].s[0];
        else if (tr[tr[u].s[0]].size + 1 == k) return tr[u].id;
        else k -= tr[tr[u].s[0]].size + 1, u = tr[u].s[1];
    }
    return -1;
}

void dfs(int u, int b)
{
    if (tr[u].s[0]) dfs(tr[u].s[0], b);
    if (tr[u].s[1]) dfs(tr[u].s[1], b);
    insert(tr[u].v, tr[u].id, b);
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ )
    {
        p[i] = root[i] = i;
        int v;
        scanf("%d", &v);
        tr[i].init(v, i, 0);
    }
    idx = n;

    while (m -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        a = find(a), b = find(b);
        if (a != b)
        {
            if (tr[root[a]].size > tr[root[b]].size) swap(a, b);
            dfs(root[a], b);
            p[a] = b;
        }
    }

    scanf("%d", &m);
    while (m -- )
    {
        char op[2];
        int a, b;
        scanf("%s%d%d", op, &a, &b);
        if (*op == 'B')
        {
            a = find(a), b = find(b);
            if (a != b)
            {
                if (tr[root[a]].size > tr[root[b]].size) swap(a, b);
                dfs(root[a], b);
                p[a] = b;
            }
        }
        else
        {
            a = find(a);
            if (tr[root[a]].size < b) puts("-1");
            else printf("%d\n", get_k(b, a));
        }
    }

    return 0;
}

#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 500010, INF = 1e9;

int n, m;
struct Node
{
    int s[2], p, v;
    int rev, same;
    int size, sum, ms, ls, rs;

    void init(int _v, int _p)
    {
        s[0] = s[1] = 0, p = _p, v = _v;
        rev = same = 0;
        size = 1, sum = ms = v;
        ls = rs = max(v, 0);
    }
}tr[N];
int root, nodes[N], tt;
int w[N];

void pushup(int x)
{
    auto &u = tr[x], &l = tr[u.s[0]], &r = tr[u.s[1]];
    u.size = l.size + r.size + 1;
    u.sum = l.sum + r.sum + u.v;
    u.ls = max(l.ls, l.sum + u.v + r.ls);
    u.rs = max(r.rs, r.sum + u.v + l.rs);
    u.ms = max(max(l.ms, r.ms), l.rs + u.v + r.ls);
}

void pushdown(int x)
{
    auto &u = tr[x], &l = tr[u.s[0]], &r = tr[u.s[1]];
    if (u.same)
    {
        u.same = u.rev = 0;
        if (u.s[0]) l.same = 1, l.v = u.v, l.sum = l.v * l.size;
        if (u.s[1]) r.same = 1, r.v = u.v, r.sum = r.v * r.size;
        if (u.v > 0)
        {
            if (u.s[0]) l.ms = l.ls = l.rs = l.sum;
            if (u.s[1]) r.ms = r.ls = r.rs = r.sum;
        }
        else
        {
            if (u.s[0]) l.ms = l.v, l.ls = l.rs = 0;
            if (u.s[1]) r.ms = r.v, r.ls = r.rs = 0;
        }
    }
    else if (u.rev)
    {
        u.rev = 0, l.rev ^= 1, r.rev ^= 1;
        swap(l.ls, l.rs), swap(r.ls, r.rs);
        swap(l.s[0], l.s[1]), swap(r.s[0], r.s[1]);
    }
}

void rotate(int x)
{
    int y = tr[x].p, z = tr[y].p;
    int k = tr[y].s[1] == x;
    tr[z].s[tr[z].s[1] == y] = x, tr[x].p = z;
    tr[y].s[k] = tr[x].s[k ^ 1], tr[tr[x].s[k ^ 1]].p = y;
    tr[x].s[k ^ 1] = y, tr[y].p = x;
    pushup(y), pushup(x);
}

void splay(int x, int k)
{
    while (tr[x].p != k)
    {
        int y = tr[x].p, z = tr[y].p;
        if (z != k)
            if ((tr[y].s[1] == x) ^ (tr[z].s[1] == y)) rotate(x);
            else rotate(y);
        rotate(x);
    }
    if (!k) root = x;
}

int get_k(int k)
{
    int u = root;
    while (u)
    {
        pushdown(u);
        if (tr[tr[u].s[0]].size >= k) u = tr[u].s[0];
        else if (tr[tr[u].s[0]].size + 1 == k) return u;
        else k -= tr[tr[u].s[0]].size + 1, u = tr[u].s[1];
    }
}

int build(int l, int r, int p)
{
    int mid = l + r >> 1;
    int u = nodes[tt -- ];
    tr[u].init(w[mid], p);
    if (l < mid) tr[u].s[0] = build(l, mid - 1, u);
    if (mid < r) tr[u].s[1] = build(mid + 1, r, u);
    pushup(u);
    return u;
}

void dfs(int u)
{
    if (tr[u].s[0]) dfs(tr[u].s[0]);
    if (tr[u].s[1]) dfs(tr[u].s[1]);
    nodes[ ++ tt] = u;
}

int main()
{
    for (int i = 1; i < N; i ++ ) nodes[ ++ tt] = i;
    scanf("%d%d", &n, &m);
    tr[0].ms = w[0] = w[n + 1] = -INF;
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);
    root = build(0, n + 1, 0);

    char op[20];
    while (m -- )
    {
        scanf("%s", op);
        if (!strcmp(op, "INSERT"))
        {
            int posi, tot;
            scanf("%d%d", &posi, &tot);
            for (int i = 0; i < tot; i ++ ) scanf("%d", &w[i]);
            int l = get_k(posi + 1), r = get_k(posi + 2);
            splay(l, 0), splay(r, l);
            int u = build(0, tot - 1, r);
            tr[r].s[0] = u;
            pushup(r), pushup(l);
        }
        else if (!strcmp(op, "DELETE"))
        {
            int posi, tot;
            scanf("%d%d", &posi, &tot);
            int l = get_k(posi), r = get_k(posi + tot + 1);
            splay(l, 0), splay(r, l);
            dfs(tr[r].s[0]);
            tr[r].s[0] = 0;
            pushup(r), pushup(l);
        }
        else if (!strcmp(op, "MAKE-SAME"))
        {
            int posi, tot, c;
            scanf("%d%d%d", &posi, &tot, &c);
            int l = get_k(posi), r = get_k(posi + tot + 1);
            splay(l, 0), splay(r, l);
            auto& son = tr[tr[r].s[0]];
            son.same = 1, son.v = c, son.sum = c * son.size;
            if (c > 0) son.ms = son.ls = son.rs = son.sum;
            else son.ms = c, son.ls = son.rs = 0;
            pushup(r), pushup(l);
        }
        else if (!strcmp(op, "REVERSE"))
        {
            int posi, tot;
            scanf("%d%d", &posi, &tot);
            int l = get_k(posi), r = get_k(posi + tot + 1);
            splay(l, 0), splay(r, l);
            auto& son = tr[tr[r].s[0]];
            son.rev ^= 1;
            swap(son.ls, son.rs);
            swap(son.s[0], son.s[1]);
            pushup(r), pushup(l);
        }
        else if (!strcmp(op, "GET-SUM"))
        {
            int posi, tot;
            scanf("%d%d", &posi, &tot);
            int l = get_k(posi), r = get_k(posi + tot + 1);
            splay(l, 0), splay(r, l);
            printf("%d\n", tr[tr[r].s[0]].sum);
        }
        else printf("%d\n", tr[root].ms);
    }

    return 0;
}

//树套树简单版
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <set>

using namespace std;

const int N = 50010, M = N * 4, INF = 1e9;

int n, m;
struct Tree
{
    int l, r;
    multiset<int> s;
}tr[M];
int w[N];

void build(int u, int l, int r)
{
    tr[u] = {l, r};
    tr[u].s.insert(-INF), tr[u].s.insert(INF);
    for (int i = l; i <= r; i ++ ) tr[u].s.insert(w[i]);
    if (l == r) return;
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
}

void change(int u, int p, int x)
{
    tr[u].s.erase(tr[u].s.find(w[p]));
    tr[u].s.insert(x);
    if (tr[u].l == tr[u].r) return;
    int mid = tr[u].l + tr[u].r >> 1;
    if (p <= mid) change(u << 1, p, x);
    else change(u << 1 | 1, p, x);
}

int query(int u, int a, int b, int x)
{
    if (tr[u].l >= a && tr[u].r <= b)
    {
        auto it = tr[u].s.lower_bound(x);
        --it;
        return *it;
    }
    int mid = tr[u].l + tr[u].r >> 1, res = -INF;
    if (a <= mid) res = max(res, query(u << 1, a, b, x));
    if (b > mid) res = max(res, query(u << 1 | 1, a, b, x));
    return res;
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);
    build (1, 1, n);

    while (m -- )
    {
        int op, a, b, x;
        scanf("%d", &op);
        if (op == 1)
        {
            scanf("%d%d", &a, &x);
            change(1, a, x);
            w[a] = x;
        }
        else
        {
            scanf("%d%d%d", &a, &b, &x);
            printf("%d\n", query(1, a, b, x));
        }
    }
    return 0;
}

//树套树
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 2000010, INF = 1e9;

int n, m;
struct Node
{
    int s[2], p, v;
    int size;

    void init(int _v, int _p)
    {
        v = _v, p = _p;
        size = 1;
    }
}tr[N];
int L[N], R[N], T[N], idx;
int w[N];

void pushup(int x)
{
    tr[x].size = tr[tr[x].s[0]].size + tr[tr[x].s[1]].size + 1;
}

void rotate(int x)
{
    int y = tr[x].p, z = tr[y].p;
    int k = tr[y].s[1] == x;
    tr[z].s[tr[z].s[1] == y] = x, tr[x].p = z;
    tr[y].s[k] = tr[x].s[k ^ 1], tr[tr[x].s[k ^ 1]].p = y;
    tr[x].s[k ^ 1] = y, tr[y].p = x;
    pushup(y), pushup(x);
}

void splay(int& root, int x, int k)
{
    while (tr[x].p != k)
    {
        int y = tr[x].p, z = tr[y].p;
        if (z != k)
            if ((tr[y].s[1] == x) ^ (tr[z].s[1] == y)) rotate(x);
            else rotate(y);
        rotate(x);
    }
    if (!k) root = x;
}

void insert(int& root, int v)
{
    int u = root, p = 0;
    while (u) p = u, u = tr[u].s[v > tr[u].v];
    u = ++ idx;
    if (p) tr[p].s[v > tr[p].v] = u;
    tr[u].init(v, p);
    splay(root, u, 0);
}

int get_k(int root, int v)
{
    int u = root, res = 0;
    while (u)
    {
        if (tr[u].v < v) res += tr[tr[u].s[0]].size + 1, u = tr[u].s[1];
        else u = tr[u].s[0];
    }
    return res;
}

void update(int& root, int x, int y)
{
    int u = root;
    while (u)
    {
        if (tr[u].v == x) break;
        if (tr[u].v < x) u = tr[u].s[1];
        else u = tr[u].s[0];
    }
    splay(root, u, 0);
    int l = tr[u].s[0], r = tr[u].s[1];
    while (tr[l].s[1]) l = tr[l].s[1];
    while (tr[r].s[0]) r = tr[r].s[0];
    splay(root, l, 0), splay(root, r, l);
    tr[r].s[0] = 0;
    pushup(r), pushup(l);
    insert(root, y);
}

void build(int u, int l, int r)
{
    L[u] = l, R[u] = r;
    insert(T[u], -INF), insert(T[u], INF);
    for (int i = l; i <= r; i ++ ) insert(T[u], w[i]);
    if (l == r) return;
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
}

int query(int u, int a, int b, int x)
{
    if (L[u] >= a && R[u] <= b) return get_k(T[u], x) - 1;
    int mid = L[u] + R[u] >> 1, res = 0;
    if (a <= mid) res += query(u << 1, a, b, x);
    if (b > mid) res += query(u << 1 | 1, a, b, x);
    return res;
}

void change(int u, int p, int x)
{
    update(T[u], w[p], x);
    if (L[u] == R[u]) return;
    int mid = L[u] + R[u] >> 1;
    if (p <= mid) change(u << 1, p, x);
    else change(u << 1 | 1, p, x);
}

int get_pre(int root, int v)
{
    int u = root, res = -INF;
    while (u)
    {
        if (tr[u].v < v) res = max(res, tr[u].v), u = tr[u].s[1];
        else u = tr[u].s[0];
    }
    return res;
}

int get_suc(int root, int v)
{
    int u = root, res = INF;
    while (u)
    {
        if (tr[u].v > v) res = min(res, tr[u].v), u = tr[u].s[0];
        else u = tr[u].s[1];
    }
    return res;
}

int query_pre(int u, int a, int b, int x)
{
    if (L[u] >= a && R[u] <= b) return get_pre(T[u], x);
    int mid = L[u] + R[u] >> 1, res = -INF;
    if (a <= mid) res = max(res, query_pre(u << 1, a, b, x));
    if (b > mid) res = max(res, query_pre(u << 1 | 1, a, b, x));
    return res;
}

int query_suc(int u, int a, int b, int x)
{
    if (L[u] >= a && R[u] <= b) return get_suc(T[u], x);
    int mid = L[u] + R[u] >> 1, res = INF;
    if (a <= mid) res = min(res, query_suc(u << 1, a, b, x));
    if (b > mid) res = min(res, query_suc(u << 1 | 1, a, b, x));
    return res;
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);
    build(1, 1, n);

    while (m -- )
    {
        int op, a, b, x;
        scanf("%d", &op);
        if (op == 1)
        {
            scanf("%d%d%d", &a, &b, &x);
            printf("%d\n", query(1, a, b, x) + 1);
        }
        else if (op == 2)
        {
            scanf("%d%d%d", &a, &b, &x);
            int l = 0, r = 1e8;
            while (l < r)
            {
                int mid = l + r + 1 >> 1;
                if (query(1, a, b, mid) + 1 <= x) l = mid;
                else r = mid - 1;
            }
            printf("%d\n", r);
        }
        else if (op == 3)
        {
            scanf("%d%d", &a, &x);
            change(1, a, x);
            w[a] = x;
        }
        else if (op == 4)
        {
            scanf("%d%d%d", &a, &b, &x);
            printf("%d\n", query_pre(1, a, b, x));
        }
        else
        {
            scanf("%d%d%d", &a, &b, &x);
            printf("%d\n", query_suc(1, a, b, x));
        }
    }

    return 0;
}

#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>

using namespace std;

typedef long long LL;

const int N = 50010, P = N * 17 * 17, M = N * 4;

int n, m;
struct Tree
{
    int l, r;
    LL sum, add;
}tr[P];
int L[M], R[M], T[M], idx;
struct Query
{
    int op, a, b, c;
}q[N];
vector<int> nums;

int get(int x)
{
    return lower_bound(nums.begin(), nums.end(), x) - nums.begin();
}

void build(int u, int l, int r)
{
    L[u] = l, R[u] = r, T[u] = ++ idx;
    if (l == r) return;
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
}

int intersection(int a, int b, int c, int d)
{
    return min(b, d) - max(a, c) + 1;
}

void update(int u, int l, int r, int pl, int pr)
{
    tr[u].sum += intersection(l, r, pl, pr);
    if (l >= pl && r <= pr)
    {
        tr[u].add ++ ;
        return;
    }
    int mid = l + r >> 1;
    if (pl <= mid)
    {
        if (!tr[u].l) tr[u].l = ++ idx;
        update(tr[u].l, l, mid, pl, pr);
    }
    if (pr > mid)
    {
        if (!tr[u].r) tr[u].r = ++ idx;
        update(tr[u].r, mid + 1, r, pl, pr);
    }
}

void change(int u, int a, int b, int c)
{
    update(T[u], 1, n, a, b);
    if (L[u] == R[u]) return;
    int mid = L[u] + R[u] >> 1;
    if (c <= mid) change(u << 1, a, b, c);
    else change(u << 1 | 1, a, b, c);
}

LL get_sum(int u, int l, int r, int pl, int pr, int add)
{
    if (l >= pl && r <= pr) return tr[u].sum + (r - l + 1LL) * add;
    int mid = l + r >> 1;
    LL res = 0;
    add += tr[u].add;
    if (pl <= mid)
    {
        if (tr[u].l) res += get_sum(tr[u].l, l, mid, pl, pr, add);
        else res += intersection(l, mid, pl, pr) * add;
    }
    if (pr > mid)
    {
        if (tr[u].r) res += get_sum(tr[u].r, mid + 1, r, pl, pr, add);
        else res += intersection(mid + 1, r, pl, pr) * add;
    }
    return res;
}

int query(int u, int a, int b, int c)
{
    if (L[u] == R[u]) return R[u];
    int mid = L[u] + R[u] >> 1;
    LL k = get_sum(T[u << 1 | 1], 1, n, a, b, 0);
    if (k >= c) return query(u << 1 | 1, a, b, c);
    return query(u << 1, a, b, c - k);
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 0; i < m; i ++ )
    {
        scanf("%d%d%d%d", &q[i].op, &q[i].a, &q[i].b, &q[i].c);
        if (q[i].op == 1) nums.push_back(q[i].c);
    }
    sort(nums.begin(), nums.end());
    nums.erase(unique(nums.begin(), nums.end()), nums.end());

    build(1, 0, nums.size() - 1);

    for (int i = 0; i < m; i ++ )
    {
        int op = q[i].op, a = q[i].a, b = q[i].b, c = q[i].c;
        if (op == 1) change(1, a, b, get(c));
        else printf("%d\n", nums[query(1, a, b, c)]);
    }

    return 0;
}


//后缀自动机
//一、SAM的性质:
//
//SAM是个状态机。一个起点，若干终点。原串的所有子串和从SAM起点开始的所有路径一一对应，不重不漏。所以终点就是包含后缀的点。
//每个点包含若干子串，每个子串都一一对应一条从起点到该点的路径。且这些子串一定是里面最长子串的连续后缀。
//SAM问题中经常考虑两种边：
//(1) 普通边，类似于Trie。表示在某个状态所表示的所有子串的后面添加一个字符。
//(2) Li缀nk、Father。表示将某个状态所表示的最短子串的首字母删除。这类边构成一棵树。
//二、SAM的构造思路
//
//endpos(s)：子串s所有出现的位置（尾字母下标）集合。SAM中的每个状态都一一对应一个endpos的等价类。
//endpos的性质：
//(1) 令 s1,s2 为 S 的两个子串 ，不妨设 |s1|≤|s2| （我们用 |s| 表示 s 的长度 ，此处等价于 s1 不长于 s2 ）。则 s1 是 s2 的后缀当且仅当 endpos(s1)⊇endpos(s2) ，s1 不是 s2 的后缀当且仅当 endpos(s1)∩endpos(s2)=∅　。
//(2) 两个不同子串的endpos，要么有包含关系，要么没有交集。
//(3) 两个子串的endpos相同，那么短串为长串的后缀。
//(4) 对于一个状态 st ，以及任意的 longest(st) 的后 s ，如果 s 的长度满足：|shortest(st)|≤|s|≤|longsest(st)| ，那么 s∈substrings(st) 。

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 2000010;

int tot = 1, last = 1;
struct Node
{
    int len, fa;
    int ch[26];
}node[N];
char str[N];
LL f[N], ans;
int h[N], e[N], ne[N], idx;

void extend(int c)
{
    int p = last, np = last = ++ tot;
    f[tot] = 1;
    node[np].len = node[p].len + 1;
    for (; p && !node[p].ch[c]; p = node[p].fa) node[p].ch[c] = np;
    if (!p) node[np].fa = 1;
    else
    {
        int q = node[p].ch[c];
        if (node[q].len == node[p].len + 1) node[np].fa = q;
        else
        {
            int nq = ++ tot;
            node[nq] = node[q], node[nq].len = node[p].len + 1;
            node[q].fa = node[np].fa = nq;
            for (; p && node[p].ch[c] == q; p = node[p].fa) node[p].ch[c] = nq;
        }
    }
}

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u)
{
    for (int i = h[u]; ~i; i = ne[i])
    {
        dfs(e[i]);
        f[u] += f[e[i]];
    }
    if (f[u] > 1) ans = max(ans, f[u] * node[u].len);
}

int main()
{
    scanf("%s", str);
    for (int i = 0; str[i]; i ++ ) extend(str[i] - 'a');
    memset(h, -1, sizeof h);
    for (int i = 2; i <= tot; i ++ ) add(node[i].fa, i);
    dfs(1);
    printf("%lld\n", ans);

    return 0;
}

#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

const int N = 10000010;

int n, m;
int tot = 1, last = 1;
char str[N];
struct Node
{
    int len, fa;
    int ch[4];
}node[N * 2];

inline int get(char c)
{
    if (c == 'E') return 0;
    if (c == 'S') return 1;
    if (c == 'W') return 2;
    return 3;
}

void extend(int c)
{
    int p = last, np = last = ++ tot;
    node[np].len = node[p].len + 1;
    for (; p && !node[p].ch[c]; p = node[p].fa) node[p].ch[c] = np;
    if (!p) node[np].fa = 1;
    else
    {
        int q = node[p].ch[c];
        if (node[q].len == node[p].len + 1) node[np].fa = q;
        else
        {
            int nq = ++ tot;
            node[nq] = node[q], node[nq].len = node[p].len + 1;
            node[q].fa = node[np].fa = nq;
            for (; p && node[p].ch[c] == q; p = node[p].fa) node[p].ch[c] = nq;
        }
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    scanf("%s", str);
    for (int i = 0; str[i]; i ++ ) extend(get(str[i]));
    while (m -- )
    {
        scanf("%s", str);
        int p = 1, res = 0;
        for (int i = 0; str[i]; i ++ )
        {
            int c = get(str[i]);
            if (node[p].ch[c]) p = node[p].ch[c], res ++ ;
            else break;
        }
        printf("%d\n", res);
    }

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 20010;

int n;
int tot = 1, last = 1;
char str[N];
struct Node
{
    int len, fa;
    int ch[26];
}node[N];
int ans[N], now[N];
int h[N], e[N], ne[N], idx;

void extend(int c)
{
    int p = last, np = last = ++ tot;
    node[np].len = node[p].len + 1;
    for (; p && !node[p].ch[c]; p = node[p].fa) node[p].ch[c] = np;
    if (!p) node[np].fa = 1;
    else
    {
        int q = node[p].ch[c];
        if (node[q].len == node[p].len + 1) node[np].fa = q;
        else
        {
            int nq = ++ tot;
            node[nq] = node[q], node[nq].len = node[p].len + 1;
            node[q].fa = node[np].fa = nq;
            for (; p && node[p].ch[c] == q; p = node[p].fa) node[p].ch[c] = nq;
        }
    }
}

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u)
{
    for (int i = h[u]; ~i; i = ne[i])
    {
        dfs(e[i]);
        now[u] = max(now[u], now[e[i]]);
    }
}

int main()
{
    scanf("%d", &n);
    scanf("%s", str);
    for (int i = 0; str[i]; i ++ ) extend(str[i] - 'a');
    for (int i = 1; i <= tot; i ++ ) ans[i] = node[i].len;
    memset(h, -1, sizeof h);
    for (int i = 2; i <= tot; i ++ ) add(node[i].fa, i);

    for (int i = 0; i < n - 1; i ++ )
    {
        scanf("%s", str);
        memset(now, 0, sizeof now);
        int p = 1, t = 0;
        for (int j = 0; str[j]; j ++ )
        {
            int c = str[j] - 'a';
            while (p > 1 && !node[p].ch[c]) p = node[p].fa, t = node[p].len;
            if (node[p].ch[c]) p = node[p].ch[c], t ++ ;
            now[p] = max(now[p], t);
        }
        dfs(1);
        for (int j = 1; j <= tot; j ++ ) ans[j] = min(ans[j], now[j]);
    }

    int res = 0;
    for (int i = 1; i <= tot; i ++ ) res = max(res, ans[i]);
    printf("%d\n", res);

    return 0;
}

//块状链表
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 2000, M = 2010;

int n, x, y;
struct Node
{
    char s[N + 1];
    int c, l, r;
}p[M];
char str[2000010];
int q[M], tt;  // 内存回收

void move(int k)  // 移到第k个字符后面
{
    x = p[0].r;
    while (k > p[x].c) k -= p[x].c, x = p[x].r;
    y = k - 1;
}

void add(int x, int u)  // 将节点u插到节点x的右边
{
    p[u].r = p[x].r, p[p[u].r].l = u;
    p[x].r = u, p[u].l = x;
}

void del(int u)  // 删除节点u
{
    p[p[u].l].r = p[u].r;
    p[p[u].r].l = p[u].l;
    p[u].l = p[u].r = p[u].c = 0;  // 清空节点u
    q[ ++ tt] = u;  // 回收节点u
}

void insert(int k)  // 在光标后插入k个字符
{
    if (y < p[x].c - 1)  // 从光标处分裂
    {
        int u = q[tt -- ];  // 新建一个节点
        for (int i = y + 1; i < p[x].c; i ++ )
            p[u].s[p[u].c ++ ] = p[x].s[i];
        p[x].c = y + 1;
        add(x, u);
    }
    int cur = x;
    for (int i = 0; i < k;)
    {
        int u = q[tt -- ];  // 创建一个新的块
        while (p[u].c < N && i < k)
            p[u].s[p[u].c ++ ] = str[i ++ ];
        add(cur, u);
        cur = u;
    }
}

void remove(int k)  // 删除光标后的k个字符
{
    if (p[x].c - 1 - y >= k)  // 节点内删
    {
        for (int i = y + k + 1, j = y + 1; i < p[x].c; i ++, j ++ ) p[x].s[j] = p[x].s[i];
        p[x].c -= k;
    }
    else
    {
        k -= p[x].c - y - 1;  // 删除当前节点的剩余部分
        p[x].c = y + 1;
        while (p[x].r && k >= p[p[x].r].c)
        {
            int u = p[x].r;
            k -= p[u].c;
            del(u);
        }
        int u = p[x].r;  // 删除结尾节点的前半部分
        for (int i = 0, j = k; j < p[u].c; i ++, j ++ ) p[u].s[i] = p[u].s[j];
        p[u].c -= k;
    }
}

void get(int k)  // 返回从光标开始的k个字符
{
    if (p[x].c - 1 - y >= k)  // 节点内返回
    {
        for (int i = 0, j = y + 1; i < k; i ++, j ++ ) putchar(p[x].s[j]);
    }
    else
    {
        k -= p[x].c - y - 1;
        for (int i = y + 1; i < p[x].c; i ++ ) putchar(p[x].s[i]);  // 输出当前节点的剩余部分
        int cur = x;
        while (p[cur].r && k >= p[p[cur].r].c)
        {
            int u = p[cur].r;
            for (int i = 0; i < p[u].c; i ++ ) putchar(p[u].s[i]);
            k -= p[u].c;
            cur = u;
        }
        int u = p[cur].r;
        for (int i = 0; i < k; i ++ ) putchar(p[u].s[i]);
    }
    puts("");
}

void prev()  // 光标向前移动一位
{
    if (!y)
    {
        x = p[x].l;
        y = p[x].c - 1;
    }
    else y -- ;
}

void next()  // 光标向后移动一位
{
    if (y < p[x].c - 1) y ++ ;
    else
    {
        x = p[x].r;
        y = 0;
    }
}

void merge()  // 将长度较短的相邻节点合并，保证块状链表时间复杂度的核心
{
    for (int i = p[0].r; i; i = p[i].r)
    {
        while (p[i].r && p[i].c + p[p[i].r].c < N)
        {
            int r = p[i].r;
            for (int j = p[i].c, k = 0; k < p[r].c; j ++, k ++ )
                p[i].s[j] = p[r].s[k];
            if (x == r) x = i, y += p[i].c;  // 更新光标的位置
            p[i].c += p[r].c;
            del(r);
        }
    }
}

int main()
{
    for (int i = 1; i < M; i ++ ) q[ ++ tt] = i;
    scanf("%d", &n);
    char op[10];

    str[0] = '>';
    insert(1);  // 插入哨兵
    move(1);  // 将光标移动到哨兵后面

    while (n -- )
    {
        int a;
        scanf("%s", op);
        if (!strcmp(op, "Move"))
        {
            scanf("%d", &a);
            move(a + 1);
        }
        else if (!strcmp(op, "Insert"))
        {
            scanf("%d", &a);
            int i = 0, k = a;
            while (a)
            {
                str[i] = getchar();
                if (str[i] >= 32 && str[i] <= 126) i ++, a -- ;
            }
            insert(k);
            merge();
        }
        else if (!strcmp(op, "Delete"))
        {
            scanf("%d", &a);
            remove(a);
            merge();
        }
        else if (!strcmp(op, "Get"))
        {
            scanf("%d", &a);
            get(a);
        }
        else if (!strcmp(op, "Prev")) prev();
        else next();
    }

    return 0;
}

//后缀数组
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1000010;

int n, m;
char s[N];
int sa[N], x[N], y[N], c[N], rk[N], height[N];

void get_sa()
{
    for (int i = 1; i <= n; i ++ ) c[x[i] = s[i]] ++ ;
    for (int i = 2; i <= m; i ++ ) c[i] += c[i - 1];
    for (int i = n; i; i -- ) sa[c[x[i]] -- ] = i;
    for (int k = 1; k <= n; k <<= 1)
    {
        int num = 0;
        for (int i = n - k + 1; i <= n; i ++ ) y[ ++ num] = i;
        for (int i = 1; i <= n; i ++ )
            if (sa[i] > k)
                y[ ++ num] = sa[i] - k;
        for (int i = 1; i <= m; i ++ ) c[i] = 0;
        for (int i = 1; i <= n; i ++ ) c[x[i]] ++ ;
        for (int i = 2; i <= m; i ++ ) c[i] += c[i - 1];
        for (int i = n; i; i -- ) sa[c[x[y[i]]] -- ] = y[i], y[i] = 0;
        swap(x, y);
        x[sa[1]] = 1, num = 1;
        for (int i = 2; i <= n; i ++ )
            x[sa[i]] = (y[sa[i]] == y[sa[i - 1]] && y[sa[i] + k] == y[sa[i - 1] + k]) ? num : ++ num;
        if (num == n) break;
        m = num;
    }
}

void get_height()
{
    for (int i = 1; i <= n; i ++ ) rk[sa[i]] = i;
    for (int i = 1, k = 0; i <= n; i ++ )
    {
        if (rk[i] == 1) continue;
        if (k) k -- ;
        int j = sa[rk[i] - 1];
        while (i + k <= n && j + k <= n && s[i + k] == s[j + k]) k ++ ;
        height[rk[i]] = k;
    }
}

int main()
{
    scanf("%s", s + 1);
    n = strlen(s + 1), m = 122;
    get_sa();
    get_height();

    for (int i = 1; i <= n; i ++ ) printf("%d ", sa[i]);
    puts("");
    for (int i = 1; i <= n; i ++ ) printf("%d ", height[i]);
    puts("");
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>

#define x first
#define y second

using namespace std;

typedef long long LL;
typedef pair<LL, LL> PLL;

const int N = 300010;
const LL INF = 2e18;

int n, m;
char s[N];
int sa[N], x[N], y[N], c[N], rk[N], height[N];
int w[N], p[N], sz[N];
LL max1[N], max2[N], min1[N], min2[N];
vector<int> hs[N];
PLL ans[N];

void get_sa()
{
    for (int i = 1; i <= n; i ++ ) c[x[i] = s[i]] ++ ;
    for (int i = 2; i <= m; i ++ ) c[i] += c[i - 1];
    for (int i = n; i; i -- ) sa[c[x[i]] -- ] = i;
    for (int k = 1; k <= n; k <<= 1)
    {
        int num = 0;
        for (int i = n - k + 1; i <= n; i ++ ) y[ ++ num] = i;
        for (int i = 1; i <= n; i ++ )
            if (sa[i] > k)
                y[ ++ num] = sa[i] - k;
        for (int i = 1; i <= m; i ++ ) c[i] = 0;
        for (int i = 1; i <= n; i ++ ) c[x[i]] ++ ;
        for (int i = 2; i <= m; i ++ ) c[i] += c[i - 1];
        for (int i = n; i; i -- ) sa[c[x[y[i]]] -- ] = y[i], y[i] = 0;
        swap(x, y);
        x[sa[1]] = 1, num = 1;
        for (int i = 2; i <= n; i ++ )
            x[sa[i]] = (y[sa[i]] == y[sa[i - 1]] && y[sa[i] + k] == y[sa[i - 1] + k]) ? num : ++ num;
        if (num == n) break;
        m = num;
    }
}

void get_height()
{
    for (int i = 1; i <= n; i ++ ) rk[sa[i]] = i;
    for (int i = 1, k = 0; i <= n; i ++ )
    {
        if (rk[i] == 1) continue;
        if (k) k -- ;
        int j = sa[rk[i] - 1];
        while (i + k <= n && j + k <= n && s[i + k] == s[j + k]) k ++ ;
        height[rk[i]] = k;
    }
}

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

LL get(int x)
{
    return x * (x - 1ll) / 2;
}

PLL calc(int r)
{
    static LL cnt = 0, maxv = -INF;

    for (auto x: hs[r])
    {
        int a = find(x - 1), b = find(x);
        cnt -= get(sz[a]) + get(sz[b]);
        p[a] = b;
        sz[b] += sz[a];
        cnt += get(sz[b]);
        if (max1[a] >= max1[b])
        {
            max2[b] = max(max1[b], max2[a]);
            max1[b] = max1[a];
        }
        else if (max1[a] > max2[b]) max2[b] = max1[a];
        if (min1[a] <= min1[b])
        {
            min2[b] = min(min1[b], min2[a]);
            min1[b] = min1[a];
        }
        else if (min1[a] < min2[b]) min2[b] = min1[a];
        maxv = max(maxv, max(max1[b] * max2[b], min1[b] * min2[b]));
    }

    if (maxv == -INF) return {cnt, 0};
    return {cnt, maxv};
}

int main()
{
    scanf("%d", &n), m = 122;
    scanf("%s", s + 1);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);

    get_sa();
    get_height();
    for (int i = 2; i <= n; i ++ ) hs[height[i]].push_back(i);

    for (int i = 1; i <= n; i ++ )
    {
        p[i] = i, sz[i] = 1;
        max1[i] = min1[i] = w[sa[i]];
        max2[i] = -INF, min2[i] = INF;
    }

    for (int i = n - 1; i >= 0; i -- ) ans[i] = calc(i);
    for (int i = 0; i < n; i ++ ) printf("%lld %lld\n", ans[i].x, ans[i].y);

    return 0;
}

//仙人掌
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 12010, M = N * 3;

int n, m, Q, new_n;
int h1[N], h2[N], e[M], w[M], ne[M], idx;
int dfn[N], low[N], cnt;
int s[N], stot[N], fu[N], fw[N], fe[N];
int fa[N][14], depth[N], d[N];
int A, B;

void add(int h[], int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void build_circle(int x, int y, int z)
{
    int sum = z;
    for (int k = y; k != x; k = fu[k])
    {
        s[k] = sum;
        sum += fw[k];
    }
    s[x] = stot[x] = sum;
    add(h2, x, ++ new_n, 0);
    for (int k = y; k != x; k = fu[k])
    {
        stot[k] = sum;
        add(h2, new_n, k, min(s[k], sum - s[k]));
    }
}

void tarjan(int u, int from)
{
    dfn[u] = low[u] = ++ cnt;
    for (int i = h1[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!dfn[j])
        {
            fu[j] = u, fw[j] = w[i], fe[j] = i;  // fe[j]存储j由哪条边下来，这样可以处理重边问题。
            tarjan(j, i);
            low[u] = min(low[u], low[j]);
            if (dfn[u] < low[j]) add(h2, u, j, w[i]);
        }
        else if (i != (from ^ 1)) low[u] = min(low[u], dfn[j]);
    }
    for (int i = h1[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (dfn[u] < dfn[j] && fe[j] != i)
            build_circle(u, j, w[i]);
    }
}

void dfs_lca(int u, int father)
{
    depth[u] = depth[father] + 1;
    fa[u][0] = father;
    for (int k = 1; k <= 13; k ++ )
        fa[u][k] = fa[fa[u][k - 1]][k - 1];
    for (int i = h2[u]; ~i; i = ne[i])
    {
        int j = e[i];
        d[j] = d[u] + w[i];
        dfs_lca(j, u);
    }
}

int lca(int a, int b)
{
    if (depth[a] < depth[b]) swap(a, b);
    for (int k = 13; k >= 0; k -- )
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if (a == b) return a;
    for (int k = 13; k >= 0; k -- )
        if (fa[a][k] != fa[b][k])
        {
            a = fa[a][k];
            b = fa[b][k];
        }
    A = a, B = b;
    return fa[a][0];
}

int main()
{
    scanf("%d%d%d", &n, &m, &Q);
    new_n = n;
    memset(h1, -1, sizeof h1);
    memset(h2, -1, sizeof h2);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(h1, a, b, c), add(h1, b, a, c);
    }
    tarjan(1, -1);

    dfs_lca(1, 0);

    while (Q -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        int p = lca(a, b);
        if (p <= n) printf("%d\n", d[a] + d[b] - d[p] * 2);
        else
        {
            int da = d[a] - d[A], db = d[b] - d[B];
            int l = abs(s[A] - s[B]);
            int dm = min(l, stot[A] - l);
            printf("%d\n", da + dm + db);
        }
    }

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 100010, M = N * 3, INF = 1e8;

int n, m, new_n;
int h1[N], h2[N], e[M], w[M], ne[M], idx;
int dfn[N], low[N], cnt;
int s[N], stot[N], fu[N], fw[N];
int d[N], f[N], q[N];
int ans;

void add(int h[], int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void build_circle(int x, int y, int z)
{
    int sum = z;
    for (int k = y; k != x; k = fu[k])
    {
        s[k] = sum;
        sum += fw[k];
    }
    s[x] = stot[x] = sum;
    add(h2, x, ++ new_n, 0);
    for (int k = y; k != x; k = fu[k])
    {
        stot[k] = sum;
        add(h2, new_n, k, min(s[k], sum - s[k]));
    }
}

void tarjan(int u, int from)
{
    dfn[u] = low[u] = ++ cnt;
    for (int i = h1[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!dfn[j])
        {
            fu[j] = u, fw[j] = w[i];
            tarjan(j, i);
            low[u] = min(low[u], low[j]);
            if (dfn[u] < low[j]) add(h2, u, j, w[i]);
        }
        else if (i != (from ^ 1)) low[u] = min(low[u], dfn[j]);
    }
    for (int i = h1[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (dfn[u] < dfn[j] && fu[j] != u)
            build_circle(u, j, w[i]);
    }
}

int dfs(int u)
{
    int d1 = 0, d2 = 0;
    for (int i = h2[u]; ~i; i = ne[i])
    {
        int j = e[i];
        int t = dfs(j) + w[i];
        if (t >= d1) d2 = d1, d1 = t;
        else if (t > d2) d2 = t;
    }
    f[u] = d1;
    if (u <= n) ans = max(ans, d1 + d2);  // u是圆点
    else  // u是方点
    {
        int sz = 0;
        d[sz ++ ] = -INF;
        for (int i = h2[u]; ~i; i = ne[i])
            d[sz ++ ] = f[e[i]];
        for (int i = 0; i < sz; i ++ ) d[sz + i] = d[i];

        int hh = 0, tt = -1;
        for (int i = 0; i < sz * 2; i ++ )
        {
            if (hh <= tt && i - q[hh] > sz / 2) hh ++ ;
            if (hh <= tt) ans = max(ans, d[i] + i + d[q[hh]] - q[hh]);
            while (hh <= tt && d[q[tt]] - q[tt] <= d[i] - i) tt -- ;
            q[ ++ tt] = i;
        }
    }

    return f[u];
}

int main()
{
    scanf("%d%d", &n, &m);
    new_n = n;
    memset(h1, -1, sizeof h1);
    memset(h2, -1, sizeof h2);
    while (m -- )
    {
        int k, x, y;
        scanf("%d%d", &k, &x);
        for (int i = 0; i < k - 1; i ++ )
        {
            scanf("%d", &y);
            add(h1, x, y, 1), add(h1, y, x, 1);
            x = y;
        }
    }
    tarjan(1, -1);
    dfs(1);

    printf("%d\n", ans);
    return 0;
}

//点分树
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 150010, M = N * 2;

int n, m, A;
int h[N], e[M], w[M], ne[M], idx;
int age[N];
bool st[N];
struct Father
{
    int u, num;
    LL dist;
};
vector<Father> f[N];
struct Son
{
    int age;
    LL dist;
    bool operator< (const Son& t) const
    {
        return age < t.age;
    }
};
vector<Son> son[N][3];

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int get_size(int u, int fa)
{
    if (st[u]) return 0;
    int res = 1;
    for (int i = h[u]; ~i; i = ne[i])
        if (e[i] != fa)
            res += get_size(e[i], u);
    return res;
}

int get_wc(int u, int fa, int tot, int& wc)
{
    if (st[u]) return 0;
    int sum = 1, ms = 0;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (j == fa) continue;
        int t = get_wc(j, u, tot, wc);
        ms = max(ms, t);
        sum += t;
    }
    ms = max(ms, tot - sum);
    if (ms <= tot / 2) wc = u;
    return sum;
}

void get_dist(int u, int fa, LL dist, int wc, int k, vector<Son>& p)
{
    if (st[u]) return;
    f[u].push_back({wc, k, dist});
    p.push_back({age[u], dist});
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (j == fa) continue;
        get_dist(j, u, dist + w[i], wc, k, p);
    }
}

void calc(int u)
{
    if (st[u]) return;
    get_wc(u, -1, get_size(u, -1), u);
    st[u] = true;

    for (int i = h[u], k = 0; ~i; i = ne[i])
    {
        int j = e[i];
        if (st[j]) continue;
        auto& p = son[u][k];
        p.push_back({-1, 0}), p.push_back({A + 1, 0});
        get_dist(j, -1, w[i], u, k, p);
        k ++ ;
        sort(p.begin(), p.end());
        for (int i = 1; i < p.size(); i ++ ) p[i].dist += p[i - 1].dist;
    }

    for (int i = h[u]; ~i; i = ne[i]) calc(e[i]);
}

LL query(int u, int l, int r)
{
    LL res = 0;
    for (auto& t: f[u])
    {
        int g = age[t.u];
        if (g >= l && g <= r) res += t.dist;
        for (int i = 0; i < 3; i ++ )
        {
            if (i == t.num) continue;
            auto& p = son[t.u][i];
            if (p.empty()) continue;
            int a = lower_bound(p.begin(), p.end(), Son({l, -1})) - p.begin();
            int b = lower_bound(p.begin(), p.end(), Son({r + 1, -1})) - p.begin();
            res += t.dist * (b - a) + p[b - 1].dist - p[a - 1].dist;
        }
    }

    for (int i = 0; i < 3; i ++ )
    {
        auto& p = son[u][i];
        if (p.empty()) continue;
        int a = lower_bound(p.begin(), p.end(), Son({l, -1})) - p.begin();
        int b = lower_bound(p.begin(), p.end(), Son({r + 1, -1})) - p.begin();
        res += p[b - 1].dist - p[a - 1].dist;
    }

    return res;
}

int main()
{
    scanf("%d%d%d", &n, &m, &A);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &age[i]);
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c), add(b, a, c);
    }
    calc(1);
    LL res = 0;
    while (m -- )
    {
        int u, a, b;
        scanf("%d%d%d", &u, &a, &b);
        int l = (a + res) % A, r = (b + res) % A;
        if (l > r) swap(l, r);
        res = query(u, l, r);
        printf("%lld\n", res);
    }

    return 0;
}

//退火
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <ctime>

#define x first
#define y second

using namespace std;

typedef pair<double, double> PDD;
const int N = 110;

int n;
PDD q[N];
double ans = 1e8;

double rand(double l, double r)
{
    return (double)rand() / RAND_MAX * (r - l) + l;
}

double get_dist(PDD a, PDD b)
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double calc(PDD p)
{
    double res = 0;
    for (int i = 0; i < n; i ++ )
        res += get_dist(p, q[i]);
    ans = min(ans, res);
    return res;
}

void simulate_anneal()
{
    PDD cur(rand(0, 10000), rand(0, 10000));
    for (double t = 1e4; t > 1e-4; t *= 0.9)
    {
        PDD np(rand(cur.x - t, cur.x + t), rand(cur.y - t, cur.y + t));
        double dt = calc(np) - calc(cur);
        if (exp(-dt / t) > rand(0, 1)) cur = np;
    }
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%lf%lf", &q[i].x, &q[i].y);

    for (int i = 0; i < 100; i ++ ) simulate_anneal();
    printf("%.0lf\n", ans);

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <ctime>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;
const int N = 55;

int n, m;
PII q[N];
int ans;

int calc()
{
    int res = 0;
    for (int i = 0; i < m; i ++ )
    {
        res += q[i].x + q[i].y;
        if (i < n)
        {
            if (q[i].x == 10) res += q[i + 1].x + q[i + 1].y;
            else if (q[i].x + q[i].y == 10)
                res += q[i + 1].x;
        }
    }
    ans = max(ans, res);
    return res;
}

void simulate_anneal()
{
    for (double t = 1e4; t > 1e-4; t *= 0.99)
    {
        int a = rand() % m, b = rand() % m;
        int x = calc();
        swap(q[a], q[b]);
        if (n + (q[n - 1].x == 10) == m)
        {
            int y = calc();
            int delta = y - x;
            if (exp(delta / t) < (double)rand() / RAND_MAX)
                swap(q[a], q[b]);
        }
        else swap(q[a], q[b]);
    }
}

int main()
{
    cin >> n;
    for (int i = 0; i < n; i ++ ) cin >> q[i].x >> q[i].y;
    if (q[n - 1].x == 10) m = n + 1, cin >> q[n].x >> q[n].y;
    else m = n;

    for (int i = 0; i < 100; i ++ ) simulate_anneal();

    cout << ans << endl;
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace std;

const int N = 25, M = 10;

int n, m;
int w[N], s[M];
double ans = 1e8;

double calc()
{
    memset(s, 0, sizeof s);
    for (int i = 0; i < n; i ++ )
    {
        int k = 0;
        for (int j = 0; j < m; j ++ )
            if (s[j] < s[k])
                k = j;
        s[k] += w[i];
    }

    double avg = 0;
    for (int i = 0; i < m; i ++ ) avg += (double)s[i] / m;
    double res = 0;
    for (int i = 0; i < m; i ++ )
        res += (s[i] - avg) * (s[i] - avg);
    res = sqrt(res / m);
    ans = min(ans, res);
    return res;
}

void simulate_anneal()
{
    random_shuffle(w, w + n);

    for (double t = 1e6; t > 1e-6; t *= 0.95)
    {
        int a = rand() % n, b = rand() % n;
        double x = calc();
        swap(w[a], w[b]);
        double y = calc();
        double delta = y - x;
        if (exp(-delta / t) < (double)rand() / RAND_MAX)
            swap(w[a], w[b]);
    }
}

int main()
{
    cin >> n >> m;
    for (int i = 0; i < n; i ++ ) cin >> w[i];

    for (int i = 0; i < 100; i ++ ) simulate_anneal();
    printf("%.2lf\n", ans);

    return 0;
}

//模拟退火
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <ctime>

#define x first
#define y second

using namespace std;

typedef pair<double, double> PDD;
const int N = 110;

int n;
PDD q[N];
double ans = 1e8;

double rand(double l, double r)
{
    return (double)rand() / RAND_MAX * (r - l) + l;
}

double get_dist(PDD a, PDD b)
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double calc(PDD p)
{
    double res = 0;
    for (int i = 0; i < n; i ++ )
        res += get_dist(p, q[i]);
    ans = min(ans, res);
    return res;
}

void simulate_anneal()
{
    PDD cur(rand(0, 10000), rand(0, 10000));
    for (double t = 1e4; t > 1e-4; t *= 0.9)
    {
        PDD np(rand(cur.x - t, cur.x + t), rand(cur.y - t, cur.y + t));
        double dt = calc(np) - calc(cur);
        if (exp(-dt / t) > rand(0, 1)) cur = np;
    }
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%lf%lf", &q[i].x, &q[i].y);

    for (int i = 0; i < 100; i ++ ) simulate_anneal();
    printf("%.0lf\n", ans);

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <ctime>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;
const int N = 55;

int n, m;
PII q[N];
int ans;

int calc()
{
    int res = 0;
    for (int i = 0; i < m; i ++ )
    {
        res += q[i].x + q[i].y;
        if (i < n)
        {
            if (q[i].x == 10) res += q[i + 1].x + q[i + 1].y;
            else if (q[i].x + q[i].y == 10)
                res += q[i + 1].x;
        }
    }
    ans = max(ans, res);
    return res;
}

void simulate_anneal()
{
    for (double t = 1e4; t > 1e-4; t *= 0.99)
    {
        int a = rand() % m, b = rand() % m;
        int x = calc();
        swap(q[a], q[b]);
        if (n + (q[n - 1].x == 10) == m)
        {
            int y = calc();
            int delta = y - x;
            if (exp(delta / t) < (double)rand() / RAND_MAX)
                swap(q[a], q[b]);
        }
        else swap(q[a], q[b]);
    }
}

int main()
{
    cin >> n;
    for (int i = 0; i < n; i ++ ) cin >> q[i].x >> q[i].y;
    if (q[n - 1].x == 10) m = n + 1, cin >> q[n].x >> q[n].y;
    else m = n;

    for (int i = 0; i < 100; i ++ ) simulate_anneal();

    cout << ans << endl;
    return 0;
}

//爬山
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace std;

const int N = 15;

int n;
double d[N][N];
double ans[N], dist[N], delta[N];

void calc()
{
    double avg = 0;
    for (int i = 0; i < n + 1; i ++ )
    {
        dist[i] = delta[i] = 0;
        for (int j = 0; j < n; j ++ )
            dist[i] += (d[i][j] - ans[j]) * (d[i][j] - ans[j]);
        dist[i] = sqrt(dist[i]);
        avg += dist[i] / (n + 1);
    }
    for (int i = 0; i < n + 1; i ++ )
        for (int j = 0; j < n; j ++ )
            delta[j] += (dist[i] - avg) * (d[i][j] - ans[j]) / avg;
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n + 1; i ++ )
        for (int j = 0; j < n; j ++ )
        {
            scanf("%lf", &d[i][j]);
            ans[j] += d[i][j] / (n + 1);
        }

    for (double t = 1e4; t > 1e-6; t *= 0.99995)
    {
        calc();
        for (int i = 0; i < n; i ++ )
            ans[i] += delta[i] * t;
    }
    for (int i = 0; i < n; i ++ ) printf("%.3lf ", ans[i]);

    return 0;
}


//网络流
//最大流知识点梳理
//1. 基本概念
//    1.1 流网络，不考虑反向边
//    1.2 可行流，不考虑反向边
//        1.2.1 两个条件：容量限制、流量守恒
//        1.2.2 可行流的流量指从源点流出的流量 - 流入源点的流量
//        1.2.3 最大流是指最大可行流
//    1.3 残留网络，考虑反向边，残留网络的可行流f' + 原图的可行流f = 原题的另一个可行流
//        (1) |f' + f| = |f'| + |f|
//        (2) |f'| 可能是负数
//    1.4 增广路径
//    1.5 割
//        1.5.1 割的定义
//        1.5.2 割的容量，不考虑反向边，“最小割”是指容量最小的割。
//        1.5.3 割的流量，考虑反向边，f(S, T) <= c(S, T)
//        1.5.4 对于任意可行流f，任意割[S, T]，|f| = f(S, T)
//        1.5.5 对于任意可行流f，任意割[S, T]，|f| <= c(S, T)
//        1.5.6 最大流最小割定理
//            (1) 可以流f是最大流
//            (2) 可行流f的残留网络中不存在增广路
//            (3) 存在某个割[S, T]，|f| = c(S, T)
//    1.6. 算法
//        1.6.1 EK O(nm^2)
//        1.6.2 Dinic O(n^2m)
//    1.7 应用
//        1.7.1 二分图
//            (1) 二分图匹配
//            (2) 二分图多重匹配
//        1.7.2 上下界网络流
//            (1) 无源汇上下界可行流
//            (2) 有源汇上下界最大流
//            (3) 有源汇上下界最小流
//        1.7.3 多源汇最大流

// EK最大流
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1010, M = 20010, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], pre[N];
bool st[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(st, false, sizeof st);
    q[0] = S, st[S] = true, d[S] = INF;
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (!st[ver] && f[i])
            {
                st[ver] = true;
                d[ver] = min(d[t], f[i]);
                pre[ver] = i;
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int EK()
{
    int r = 0;
    while (bfs())
    {
        r += d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
            f[pre[i]] -= d[T], f[pre[i] ^ 1] += d[T];
    }
    return r;
}

int main()
{
    scanf("%d%d%d%d", &n, &m, &S, &T);
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    printf("%d\n", EK());

    return 0;
}

//Dinic/ISAP求最大流
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 10010, M = 200010, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T)  return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;  // 当前弧优化
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d%d%d", &n, &m, &S, &T);
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    printf("%d\n", dinic());

    return 0;
}

//最大流之二分图匹配
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 110, M = 5210, INF = 1e8;

int m, n, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &m, &n);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= m; i ++ ) add(S, i, 1);
    for (int i = m + 1; i <= n; i ++ ) add(i, T, 1);

    int a, b;
    while (cin >> a >> b, a != -1) add(a, b, 1);

    printf("%d\n", dinic());
    for (int i = 0; i < idx; i += 2)
        if (e[i] > m && e[i] <= n && !f[i])
            printf("%d %d\n", e[i ^ 1], e[i]);

    return 0;
}

#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 430, M = (150 * 270 + N) * 2, INF = 1e8;

int m, n, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &m, &n);
    S = 0, T = m + n + 1;
    memset(h, -1, sizeof h);

    int tot = 0;
    for (int i = 1; i <= m; i ++ )
    {
        int c;
        scanf("%d", &c);
        add(S, i, c);
        tot += c;
    }
    for (int i = 1; i <= n; i ++ )
    {
        int c;
        scanf("%d", &c);
        add(m + i, T, c);
    }
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= n; j ++ )
            add(i, m + j, 1);

    if (dinic() != tot) puts("0");
    else
    {
        puts("1");
        for (int i = 1; i <= m; i ++ )
        {
            for (int j = h[i]; ~j; j = ne[j])
                if (e[j] > m && e[j] <= m + n && !f[j])
                    printf("%d ", e[j] - m);
            puts("");
        }
    }

    return 0;
}

////无源汇上下界可行流
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 210, M = (10200 + N) * 2, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], l[M], ne[M], idx;
int q[N], d[N], cur[N], A[N];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = d - c, l[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    for (int i = 0; i < m; i ++ )
    {
        int a, b, c, d;
        scanf("%d%d%d%d", &a, &b, &c, &d);
        add(a, b, c, d);
        A[a] -= c, A[b] += c;
    }

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
        if (A[i] > 0) add(S, i, 0, A[i]), tot += A[i];
        else if (A[i] < 0) add(i, T, 0, -A[i]);

    if (dinic() != tot) puts("NO");
    else
    {
        puts("YES");
        for (int i = 0; i < m * 2; i += 2)
            printf("%d\n", f[i ^ 1] + l[i]);
    }
    return 0;
}

//有源汇上下界最大流
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 210, M = (N + 10000) * 2, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N], A[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    int s, t;
    scanf("%d%d%d%d", &n, &m, &s, &t);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c, d;
        scanf("%d%d%d%d", &a, &b, &c, &d);
        add(a, b, d - c);
        A[a] -= c, A[b] += c;
    }

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
        if (A[i] > 0) add(S, i, A[i]), tot += A[i];
        else if (A[i] < 0) add(i, T, -A[i]);

    add(t, s, INF);
    if (dinic() < tot) puts("No Solution");
    else
    {
        int res = f[idx - 1];
        S = s, T = t;
        f[idx - 1] = f[idx - 2] = 0;
        printf("%d\n", res + dinic());
    }

    return 0;
}

//有源汇上下界最小流
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 50010, M = (N + 125003) * 2, INF = 2147483647;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N], A[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    int s, t;
    scanf("%d%d%d%d", &n, &m, &s, &t);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c, d;
        scanf("%d%d%d%d", &a, &b, &c, &d);
        add(a, b, d - c);
        A[a] -= c, A[b] += c;
    }

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
        if (A[i] > 0) add(S, i, A[i]), tot += A[i];
        else if (A[i] < 0) add(i, T, -A[i]);

    add(t, s, INF);

    if (dinic() < tot) puts("No Solution");
    else
    {
        int res = f[idx - 1];
        S = t, T = s;
        f[idx - 1] = f[idx - 2] = 0;
        printf("%d\n", res - dinic());
    }

    return 0;
}

//多源汇最大流
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 10010, M = (100000 + N) * 2, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    int sc, tc;
    scanf("%d%d%d%d", &n, &m, &sc, &tc);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    while (sc -- )
    {
        int x;
        scanf("%d", &x);
        add(S, x, INF);
    }
    while (tc -- )
    {
        int x;
        scanf("%d", &x);
        add(x, T, INF);
    }

    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    printf("%d\n", dinic());
    return 0;
}

//最大流之关键边
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 510, M = 10010, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
bool vis_s[N], vis_t[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

void dfs(int u, bool st[], int t)
{
    st[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = i ^ t, ver = e[i];
        if (f[j] && !st[ver])
            dfs(ver, st, t);
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n - 1;
    memset(h, -1, sizeof h);
    for (int i = 0; i < m; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    dinic();
    dfs(S, vis_s, 0);
    dfs(T, vis_t, 1);

    int res = 0;
    for (int i = 0; i < m * 2; i += 2)
        if (!f[i] && vis_s[e[i ^ 1]] && vis_t[e[i]])
            res ++ ;

    printf("%d\n", res);
    return 0;
}

//最大流之最大流判定
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 210, M = 80010, INF = 1e8;

int n, m, K, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, w[idx] = c, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

bool check(int mid)
{
    for (int i = 0; i < idx; i ++ )
        if (w[i] > mid) f[i] = 0;
        else f[i] = 1;

    return dinic() >= K;
}

int main()
{
    scanf("%d%d%d", &n, &m, &K);
    S = 1, T = n;
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    int l = 1, r = 1e6;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;
        else l = mid + 1;
    }

    printf("%d\n", r);

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1101 * 50 + 10, M = (N + 1100 + 20 * 1101) + 10, INF = 1e8;

int n, m, k, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
struct Ship
{
    int h, r, id[30];
}ships[30];
int p[30];

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int get(int i, int day)
{
    return day * (n + 2) + i;
}

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d%d", &n, &m, &k);
    S = N - 2, T = N - 1;
    memset(h, -1, sizeof h);
    for (int i = 0; i < 30; i ++ ) p[i] = i;
    for (int i = 0; i < m; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        ships[i] = {a, b};
        for (int j = 0; j < b; j ++ )
        {
            int id;
            scanf("%d", &id);
            if (id == -1) id = n + 1;
            ships[i].id[j] = id;
            if (j)
            {
                int x = ships[i].id[j - 1];
                p[find(x)] = find(id);
            }
        }
    }
    if (find(0) != find(n + 1)) puts("0");
    else
    {
        add(S, get(0, 0), k);
        add(get(n + 1, 0), T, INF);
        int day = 1, res = 0;
        while (true)
        {
            add(get(n + 1, day), T, INF);
            for (int i = 0; i <= n + 1; i ++ )
                add(get(i, day - 1), get(i, day), INF);
            for (int i = 0; i < m; i ++ )
            {
                int r = ships[i].r;
                int a = ships[i].id[(day - 1) % r], b = ships[i].id[day % r];
                add(get(a, day - 1), get(b, day), ships[i].h);
            }
            res += dinic();
            if (res >= k) break;
            day ++ ;
        }

        printf("%d\n", day);
    }

    return 0;
}

//最大流之拆点
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 410, M = 40610, INF = 1e8;

int n, F, D, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d%d", &n, &F, &D);
    S = 0, T = n * 2 + F + D + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= F; i ++ ) add(S, n * 2 + i, 1);
    for (int i = 1; i <= D; i ++ ) add(n * 2 + F + i, T, 1);
    for (int i = 1; i <= n; i ++ )
    {
        add(i, n + i, 1);
        int a, b, t;
        scanf("%d%d", &a, &b);
        while (a -- )
        {
            scanf("%d", &t);
            add(n * 2 + t, i, 1);
        }
        while (b -- )
        {
            scanf("%d", &t);
            add(i + n, n * 2 + F + t, 1);
        }
    }
    printf("%d\n", dinic());
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1010, M = 251010, INF = 1e8;

int n, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
int g[N], w[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d", &n);
    S = 0, T = n * 2 + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);
    int s = 0;
    for (int i = 1; i <= n; i ++ )
    {
        add(i, i + n, 1);
        g[i] = 1;
        for (int j = 1; j < i; j ++ )
            if (w[j] <= w[i])
                g[i] = max(g[i], g[j] + 1);
        for (int j = 1; j < i; j ++ )
            if (w[j] <= w[i] && g[j] + 1 == g[i])
                add(n + j, i, 1);
        s = max(s, g[i]);
        if (g[i] == 1) add(S, i, 1);
    }

    for (int i = 1; i <= n; i ++ )
        if (g[i] == s)
            add(n + i, T, 1);

    printf("%d\n", s);
    if (s == 1) printf("%d\n%d\n", n, n);
    else
    {
        int res = dinic();
        printf("%d\n", res);
        for (int i = 0; i < idx; i += 2)
        {
            int a = e[i ^ 1], b = e[i];
            if (a == S && b == 1) f[i] = INF;
            else if (a == 1 && b == n + 1) f[i] = INF;
            else if (a == n && b == n + n) f[i] = INF;
            else if (a == n + n && b == T) f[i] = INF;
        }
        printf("%d\n", res + dinic());
    }

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 210, M = 20410, INF = 1e8;
const double eps = 1e-8;

int n, S, T;
double D;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
PII p[N];

bool check(PII a, PII b)
{
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy < D * D + eps;
}

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    int cases;
    scanf("%d", &cases);
    while (cases -- )
    {
        memset(h, -1, sizeof h);
        idx = 0;
        scanf("%d%lf", &n, &D);
        S = 0;

        int tot = 0;
        for (int i = 1; i <= n; i ++ )
        {
            int x, y, a, b;
            scanf("%d%d%d%d", &x, &y, &a, &b);
            p[i] = {x, y};
            add(S, i, a);
            add(i, n + i, b);
            tot += a;
        }

        for (int i = 1; i <= n; i ++ )
            for (int j = i + 1; j <= n; j ++ )
                if (check(p[i], p[j]))
                {
                    add(n + i, j, INF);
                    add(n + j, i, INF);
                }

        int cnt = 0;
        for (int i = 1; i <= n; i ++ )
        {
            T = i;
            for (int j = 0; j < idx; j += 2)
            {
                f[j] += f[j ^ 1];
                f[j ^ 1] = 0;
            }
            if (dinic() == tot)
            {
                printf("%d ", i - 1);
                cnt ++ ;
            }
        }
        if (!cnt) puts("-1");
        else puts("");
    }

    return 0;
}

//最大流之建图实战
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 110, M = 100200 * 2 + 10, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
int w[1010], belong[1010];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs())
    {
        r += find(S, INF);
        flow = find(S, INF);
        if (flow) puts("!");
        r += flow;
    }
    return r;
}

int main()
{
    scanf("%d%d", &m, &n);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= m; i ++ ) scanf("%d", &w[i]);
    for (int i = 1; i <= n; i ++ )
    {
        int a, b;
        scanf("%d", &a);
        while (a -- )
        {
            int t;
            scanf("%d", &t);
            if (!belong[t]) add(S, i, w[t]);
            else add(belong[t], i, INF);
            belong[t] = i;
        }
        scanf("%d", &b);
        add(i, T, b);
    }

    printf("%d\n", dinic());
    return 0;
}

//最小割之算法模板
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 10010, M = 200010, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d%d%d", &n, &m, &S, &T);
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }
    printf("%d\n", dinic());
    return 0;
}

//最小割之直接应用
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 110, M = 810, INF = 1e8;
const double eps = 1e-8;

int n, m, S, T;
int h[N], e[M], w[M], ne[M], idx;
double f[M];
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, w[idx] = c, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i] > 0)
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

double find(int u, double limit)
{
    if (u == T) return limit;
    double flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i] > 0)
        {
            double t = find(ver, min(f[i], limit - flow));
            if (t < eps) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

double dinic(double mid)
{
    double res = 0;
    for (int i = 0; i < idx; i += 2)
        if (w[i] <= mid)
        {
            res += w[i] - mid;
            f[i] = f[i ^ 1] = 0;
        }
        else f[i] = f[i ^ 1] = w[i] - mid;

    double r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r + res;
}

int main()
{
    scanf("%d%d%d%d", &n, &m, &S, &T);
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    double l = 0, r = 1e7;
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (dinic(mid) < 0) r = mid;
        else l = mid;
    }

    printf("%.2lf\n", r);
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef long long LL;
typedef pair<int, int> PII;

const int N = 510, M = (3000 + N * 2) * 2, INF = 1e8;

int n, m, k, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
int p[N];
PII edges[3010];

void add(int a, int b, int c1, int c2)
{
    e[idx] = b, f[idx] = c1, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = c2, ne[idx] = h[b], h[b] = idx ++ ;
}

void build(int k)
{
    memset(h, -1, sizeof h);
    idx = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = edges[i].x, b = edges[i].y;
        add(a, b, 1, 1);
    }
    for (int i = 1; i <= n; i ++ )
        if (p[i] >= 0)
        {
            if (p[i] >> k & 1) add(i, T, INF, 0);
            else add(S, i, INF, 0);
        }
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

LL dinic(int k)
{
    build(k);
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n + 1;
    for (int i = 0; i < m; i ++ ) scanf("%d%d", &edges[i].x, &edges[i].y);
    scanf("%d", &k);
    memset(p, -1, sizeof p);
    while (k -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        p[a] = b;
    }

    LL res = 0;
    for (int i = 0; i <= 30; i ++ ) res += dinic(i) << i;
    printf("%lld\n", res);

    return 0;
}

//最小割之平面图转最短路
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 5050;
const LL INF = 0x3f3f3f3f3f3f3f3fLL;

int n, m, A, B, Q, X;
int r[N][N], c[N][N];
LL d[N];

int main()
{
    scanf("%d%d%d%d%d%d", &n, &m, &A, &B, &Q, &X);
    for (int i = 1; i <= n - 1; i ++ )
        for (int j = 1; j <= m; j ++ )
            c[i][j] = X = ((LL)A * X + B) % Q;
    for (int i = 2; i <= n; i ++ )
        for (int j = 1; j < m; j ++ )
            r[i][j] = X = ((LL)A * X + B) % Q;

    for (int j = 1; j <= m; j ++ )
    {
        for (int i = 1; i <= n - 1; i ++ )
            d[i] += c[i][j];
        for (int i = 2; i <= n - 1; i ++ )
            d[i] = min(d[i], d[i - 1] + r[i][j]);
        for (int i = n - 2; i; i -- )
            d[i] = min(d[i], d[i + 1] + r[i + 1][j]);
    }
    LL res = INF;
    for (int i = 1; i <= n - 1; i ++ ) res = min(res, d[i]);
    printf("%lld\n", res);
    return 0;
}

//最小割之最大权闭合图
//最大权闭合图
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 55010, M = (50000 * 3 + 5000) * 2 + 10, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n + m + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ )
    {
        int p;
        scanf("%d", &p);
        add(m + i, T, p);
    }

    int tot = 0;
    for (int i = 1; i <= m; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(S, i, c);
        add(i, m + a, INF);
        add(i, m + b, INF);
        tot += c;
    }

    printf("%d\n", tot - dinic());

    return 0;
}

//最小割之最大密度子图
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 5010, M = (50000 + N * 2) * 2 + 10, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
int dg[N], p[N];

void add(int a, int b, int c1, int c2)
{
    e[idx] = b, f[idx] = c1, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = c2, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &p[i]), p[i] *= -1;
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c, c);
        dg[a] += c, dg[b] += c;
    }
    int U = 0;
    for (int i = 1; i <= n; i ++ ) U = max(U, 2 * p[i] + dg[i]);
    for (int i = 1; i <= n; i ++ )
    {
        add(S, i, U, 0);
        add(i, T, U - 2 * p[i] - dg[i], 0);
    }

    printf("%d\n", (U * n - dinic()) / 2);

    return 0;
}

//最小割之最小点权覆盖集
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 210, M = 5200 * 2 + 10, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
bool st[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

void dfs(int u)
{
    st[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
        if (f[i] && !st[e[i]])
            dfs(e[i]);
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n * 2 + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ )
    {
        int w;
        scanf("%d",  &w);
        add(S, i, w);
    }
    for (int i = 1; i <= n; i ++ )
    {
        int w;
        scanf("%d", &w);
        add(n + i, T, w);
    }

    while (m -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(b, n + a, INF);
    }

    printf("%d\n", dinic());
    dfs(S);

    int cnt = 0;
    for (int i = 0; i < idx; i += 2)
    {
        int a = e[i ^ 1], b = e[i];
        if (st[a] && !st[b]) cnt ++ ;
    }

    printf("%d\n", cnt);
    for (int i = 0; i < idx; i += 2)
    {
        int a = e[i ^ 1], b = e[i];
        if (st[a] && !st[b])
        {
            if (a == S) printf("%d +\n", b);
        }
    }
    for (int i = 0; i < idx; i += 2)
    {
        int a = e[i ^ 1], b = e[i];
        if (st[a] && !st[b])
        {
            if (b == T) printf("%d -\n", a - n);
        }
    }

    return 0;
}

//最小割之最大点权独立集
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 10010, M = 60010, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

int get(int x, int y)
{
    return (x - 1) * m + y;
}

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n * m + 1;
    memset(h, -1, sizeof h);

    int dx[] = {-1, 0, 1, 0}, dy[] = {0, 1, 0, -1};

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
        {
            int w;
            scanf("%d", &w);
            if (i + j & 1)
            {
                add(S, get(i, j), w);
                for (int k = 0; k < 4; k ++ )
                {
                    int x = i + dx[k], y = j + dy[k];
                    if (x >= 1 && x <= n && y >= 1 && y <= m)
                        add(get(i, j), get(x, y), INF);
                }
            }
            else
                add(get(i, j), T, w);
            tot += w;
        }

    printf("%d\n", tot - dinic());
    return 0;
}

//最小割之建图实战
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 110, M = 5210, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    while (cin >> n >> m)
    {
        memset(h, -1, sizeof h);
        idx = 0;
        for (int i = 0; i < n; i ++ ) add(i, n + i, 1);
        while (m -- )
        {
            int a, b;
            scanf(" (%d,%d)", &a, &b);
            add(n + a, b, INF);
            add(n + b, a, INF);
        }
        int res = n;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < i; j ++ )
            {
                S = n + i, T = j;
                for (int k = 0; k < idx; k += 2)
                {
                    f[k] += f[k ^ 1];
                    f[k ^ 1] = 0;
                }
                res = min(res, dinic());
            }
        printf("%d\n", res);
    }

    return 0;
}

#include <iostream>
#include <cstring>
#include <sstream>
#include <algorithm>

using namespace std;

const int N = 110, M = 5210, INF = 1e8;

int m, n, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
bool st[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

void dfs(int u)
{
    st[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
        if (!st[e[i]] && f[i])
            dfs(e[i]);
}

int main()
{
    scanf("%d%d", &m, &n);
    S = 0, T = m + n + 1;
    memset(h, -1, sizeof h);
    getchar();  // 过滤掉第一行最后的回程

    int tot = 0;
    for (int i = 1; i <= m; i ++ )
    {
        int w, id;
        string line;
        getline(cin, line);
        stringstream ssin(line);
        ssin >> w;
        add(S, i, w);
        while (ssin >> id) add(i, m + id, INF);
        tot += w;
    }
    for (int i = 1; i <= n; i ++ )
    {
        int p;
        cin >> p;
        add(m + i, T, p);
    }

    int res = dinic();
    dfs(S);

    for (int i = 1; i <= m; i ++ )
        if (st[i]) printf("%d ", i);
    puts("");
    for (int i = m + 1; i <= m + n; i ++ )
        if (st[i]) printf("%d ", i - m);
    puts("");
    printf("%d\n", tot - res);
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 40010, M = 400010, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];
bool g[210][210];

int get(int x, int y)
{
    return (x - 1) * n + y;
}

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n * n + 1;
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int x, y;
        scanf("%d%d", &x, &y);
        g[x][y] = true;
    }

    int dx[] = {-2, -1, 1, 2, 2, 1, -1, -2};
    int dy[] = {1, 2, 2, 1, -1, -2, -2, -1};

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
        {
            if (g[i][j]) continue;
            if (i + j & 1)
            {
                add(S, get(i, j), 1);
                for (int k = 0; k < 8; k ++ )
                {
                    int x = i + dx[k], y = j + dy[k];
                    if (x >= 1 && x <= n && y >= 1 && y <= n && !g[x][y])
                        add(get(i, j), get(x, y), INF);
                }
            }
            else add(get(i, j), T, 1);
            tot ++ ;
        }

    printf("%d\n", tot - dinic());
    return 0;
}

//费用流之算法模板
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 5010, M = 100010, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, 0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;

        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] > d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(f[i], incf[t]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }

    return incf[T] > 0;
}

void EK(int& flow, int& cost)
{
    flow = cost = 0;
    while (spfa())
    {
        int t = incf[T];
        flow += t, cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
}

int main()
{
    scanf("%d%d%d%d", &n, &m, &S, &T);
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c, d;
        scanf("%d%d%d%d", &a, &b, &c, &d);
        add(a, b, c, d);
    }

    int flow, cost;
    EK(flow, cost);
    printf("%d %d\n", flow, cost);

    return 0;
}

//费用流之直接应用
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 160, M = 5150 * 2 + 10, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, 0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] > d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(incf[t], f[i]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }
    return incf[T] > 0;
}

int EK()
{
    int cost = 0;
    while (spfa())
    {
        int t = incf[T];
        cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
    return cost;
}

int main()
{
    scanf("%d%d", &m, &n);
    S = 0, T = m + n + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= m; i ++ )
    {
        int a;
        scanf("%d", &a);
        add(S, i, a, 0);
    }
    for (int i = 1; i <= n; i ++ )
    {
        int b;
        scanf("%d", &b);
        add(m + i, T, b, 0);
    }
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= n; j ++ )
        {
            int c;
            scanf("%d", &c);
            add(i, m + j, INF, c);
        }

    printf("%d\n", EK());

    for (int i = 0; i < idx; i += 2)
    {
        f[i] += f[i ^ 1], f[i ^ 1] = 0;
        w[i] = -w[i], w[i ^ 1] = -w[i ^ 1];
    }
    printf("%d\n", -EK());

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 110, M = 610, INF = 1e8;

int n, S, T;
int s[N];
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, 0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;

        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] > d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(f[i], incf[t]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }
    return incf[T] > 0;
}

int EK()
{
    int cost = 0;
    while (spfa())
    {
        int t = incf[T];
        cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
    return cost;
}

int main()
{
    scanf("%d", &n);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
    {
        scanf("%d", &s[i]);
        tot += s[i];
        add(i, i < n ? i + 1 : 1, INF, 1);
        add(i, i > 1 ? i - 1 : n, INF, 1);
    }

    tot /= n;
    for (int i = 1; i <= n; i ++ )
        if (tot < s[i])
            add(S, i, s[i] - tot, 0);
        else if (tot > s[i])
            add(i, T, tot - s[i], 0);

    printf("%d\n", EK());
    return 0;
}

//费用流之二分图最优匹配
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 110, M = 5210, INF = 1e8;

int n, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, 0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] > d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(f[i], incf[t]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }
    return incf[T] > 0;
}

int EK()
{
    int cost = 0;
    while (spfa())
    {
        int t = incf[T];
        cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
    return cost;
}

int main()
{
    scanf("%d", &n);
    S = 0, T = n * 2 + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ )
    {
        add(S, i, 1, 0);
        add(n + i, T, 1, 0);
    }
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
        {
            int c;
            scanf("%d", &c);
            add(i, n + j, 1, c);
        }

    printf("%d\n", EK());

    for (int i = 0; i < idx; i += 2)
    {
        f[i] += f[i ^ 1], f[i ^ 1] = 0;
        w[i] = -w[i], w[i ^ 1] = -w[i ^ 1];
    }
    printf("%d\n", -EK());

    return 0;
}

//费用流之最大权不相交路径
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1200, M = 4000, INF = 1e8;

int m, n, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];
int id[40][40], cost[40][40];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, -0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;

        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] < d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(f[i], incf[t]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }
    return incf[T] > 0;
}

int EK()
{
    int cost = 0;
    while (spfa())
    {
        int t = incf[T];
        cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
    return cost;
}

int main()
{
    int cnt = 0;
    scanf("%d%d", &m, &n);
    S = ++ cnt;
    T = ++ cnt;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m + i - 1; j ++ )
        {
            scanf("%d", &cost[i][j]);
            id[i][j] = ++ cnt;
        }

    // 规则1
    memset(h, -1, sizeof h), idx = 0;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m + i - 1; j ++ )
        {
            add(id[i][j] * 2, id[i][j] * 2 + 1, 1, cost[i][j]);
            if (i == 1) add(S, id[i][j] * 2, 1, 0);
            if (i == n) add(id[i][j] * 2 + 1, T, 1, 0);
            if (i < n)
            {
                add(id[i][j] * 2 + 1, id[i + 1][j] * 2, 1, 0);
                add(id[i][j] * 2 + 1, id[i + 1][j + 1] * 2, 1, 0);
            }
        }
    printf("%d\n", EK());

    // 规则2
    memset(h, -1, sizeof h), idx = 0;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m + i - 1; j ++ )
        {
            add(id[i][j] * 2, id[i][j] * 2 + 1, INF, cost[i][j]);
            if (i == 1) add(S, id[i][j] * 2, 1, 0);
            if (i == n) add(id[i][j] * 2 + 1, T, INF, 0);
            if (i < n)
            {
                add(id[i][j] * 2 + 1, id[i + 1][j] * 2, 1, 0);
                add(id[i][j] * 2 + 1, id[i + 1][j + 1] * 2, 1, 0);
            }
        }
    printf("%d\n", EK());

    // 规则3
    memset(h, -1, sizeof h), idx = 0;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m + i - 1; j ++ )
        {
            add(id[i][j] * 2, id[i][j] * 2 + 1, INF, cost[i][j]);
            if (i == 1) add(S, id[i][j] * 2, 1, 0);
            if (i == n) add(id[i][j] * 2 + 1, T, INF, 0);
            if (i < n)
            {
                add(id[i][j] * 2 + 1, id[i + 1][j] * 2, INF, 0);
                add(id[i][j] * 2 + 1, id[i + 1][j + 1] * 2, INF, 0);
            }
        }
    printf("%d\n", EK());

    return 0;
}

//费用流之网格图模型
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 5010, M = 20010, INF = 1e8;

int n, k, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];

int get(int x, int y, int t)
{
    return (x * n + y) * 2 + t;
}

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, -0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;

        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] < d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(incf[t], f[i]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }
    return incf[T] > 0;
}

int EK()
{
    int cost = 0;
    while (spfa())
    {
        int t = incf[T];
        cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
    return cost;
}

int main()
{
    scanf("%d%d", &n, &k);
    S = 2 * n * n, T = S + 1;
    memset(h, -1, sizeof h);
    add(S, get(0, 0, 0), k, 0);
    add(get(n - 1, n - 1, 1), T, k, 0);
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ )
        {
            int c;
            scanf("%d", &c);
            add(get(i, j, 0), get(i, j, 1), 1, c);
            add(get(i, j, 0), get(i, j, 1), INF, 0);
            if (i + 1 < n) add(get(i, j, 1), get(i + 1, j, 0), INF, 0);
            if (j + 1 < n) add(get(i, j, 1), get(i, j + 1, 0), INF, 0);
        }

    printf("%d\n", EK());

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 260, M = 2000, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];

int get(int x, int y)
{
    return x * (m + 1) + y;
}

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, -0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] < d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(f[i], incf[t]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }
    return incf[T] > 0;
}

int EK()
{
    int cost = 0;
    while (spfa())
    {
        int t = incf[T];
        cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
    return cost;
}

int main()
{
    int A, B;
    scanf("%d%d%d%d", &A, &B, &n, &m);
    S = (n + 1) * (m + 1), T = S + 1;
    memset(h, -1, sizeof h);
    for (int i = 0; i <= n; i ++ )
        for (int j = 0; j < m; j ++ )
        {
            int c;
            scanf("%d", &c);
            add(get(i, j), get(i, j + 1), 1, c);
            add(get(i, j), get(i, j + 1), INF, 0);
        }
    for (int i = 0; i <= m; i ++ )
        for (int j = 0; j < n; j ++ )
        {
            int c;
            scanf("%d", &c);
            add(get(j, i), get(j + 1, i), 1, c);
            add(get(j, i), get(j + 1, i), INF, 0);
        }
    while (A -- )
    {
        int k, x, y;
        scanf("%d%d%d", &k, &x, &y);
        add(S, get(x, y), k, 0);
    }
    while (B -- )
    {
        int r, x, y;
        scanf("%d%d%d", &r, &x, &y);
        add(get(x, y), T, r, 0);
    }

    printf("%d\n", EK());

    return 0;
}

//费用流之拆点
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1610, M = 10000, INF = 1e8;

int n, p, x, xp, y, yp, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, 0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] > d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(f[i], incf[t]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }
    return incf[T] > 0;
}

int EK()
{
    int cost = 0;
    while (spfa())
    {
        int t = incf[T];
        cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
    return cost;
}

int main()
{
    scanf("%d%d%d%d%d%d", &n, &p, &x, &xp, &y, &yp);
    S = 0, T = n * 2 + 1;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ )
    {
        int r;
        scanf("%d", &r);
        add(S, i, r, 0);
        add(n + i, T, r, 0);
        add(S, n + i, INF, p);
        if (i < n) add(i, i + 1, INF, 0);
        if (i + x <= n) add(i, n + i + x, INF, xp);
        if (i + y <= n) add(i, n + i + y, INF, yp);
    }
    printf("%d\n", EK());
    return 0;
}

//费用流之上下界可行流
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1010, M = 24010, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], w[M], ne[M], idx;
int q[N], d[N], pre[N], incf[N];
bool st[N];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx ++ ;
}

bool spfa()
{
    int hh = 0, tt = 1;
    memset(d, 0x3f, sizeof d);
    memset(incf, 0, sizeof incf);
    q[0] = S, d[S] = 0, incf[S] = INF;
    while (hh != tt)
    {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;

        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (f[i] && d[ver] > d[t] + w[i])
            {
                d[ver] = d[t] + w[i];
                pre[ver] = i;
                incf[ver] = min(f[i], incf[t]);
                if (!st[ver])
                {
                    q[tt ++ ] = ver;
                    if (tt == N) tt = 0;
                    st[ver] = true;
                }
            }
        }
    }
    return incf[T] > 0;
}

int EK()
{
    int cost = 0;
    while (spfa())
    {
        int t = incf[T];
        cost += t * d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
    return cost;
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n + 2;
    memset(h, -1, sizeof h);
    int last = 0;
    for (int i = 1; i <= n; i ++ )
    {
        int cur;
        scanf("%d", &cur);
        if (last > cur) add(S, i, last - cur, 0);
        else if (last < cur) add(i, T, cur - last, 0);
        add(i, i + 1, INF - cur, 0);
        last = cur;
    }
    add(S, n + 1, last, 0);

    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(b + 1, a, INF, c);
    }

    printf("%d\n", EK());
    return 0;
}

//2-SAT
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdio>

using namespace std;

const int N = 2000010, M = 2000010;

int n, m;
int h[N], e[M], ne[M], idx;
int dfn[N], low[N], ts, stk[N], top;
int id[N], cnt;
bool ins[N];

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void tarjan(int u)
{
    dfn[u] = low[u] = ++ ts;
    stk[ ++ top] = u, ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u] = min(low[u], low[j]);
        } else if (ins[j]) low[u] = min(low[u], dfn[j]);
    }

    if (low[u] == dfn[u])
    {
        int y;
        cnt ++ ;
        do
        {
            y = stk[top -- ], ins[y] = false, id[y] = cnt;
        } while (y != u);
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);

    while (m -- )
    {
        int i, a, j, b;
        scanf("%d%d%d%d", &i, &a, &j, &b);
        i --, j -- ;
        add(2 * i + !a, 2 * j + b);
        add(2 * j + !b, 2 * i + a);
    }

    for (int i = 0; i < n * 2; i ++ )
        if (!dfn[i])
            tarjan(i);

    for (int i = 0; i < n; i ++ )
        if (id[i * 2] == id[i * 2 + 1])
        {
            puts("IMPOSSIBLE");
            return 0;
        }

    puts("POSSIBLE");
    for (int i = 0; i < n; i ++ )
        if (id[i * 2] < id[i * 2 + 1]) printf("0 ");
        else printf("1 ");

    return 0;
}

#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 2010, M = 4000010;

int n;
int h[N], e[M], ne[M], idx;
int dfn[N], low[N], ts, stk[N], top;
int id[N], cnt;
bool ins[N];

struct Wedding
{
    int s, t, d;
}w[N];

bool is_overlap(int a, int b, int c, int d)
{
    return d > a && b > c;
}

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void tarjan(int u)
{
    dfn[u] = low[u] = ++ ts;
    stk[ ++ top] = u, ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u] = min(low[u], low[j]);
        } else if (ins[j]) low[u] = min(low[u], dfn[j]);
    }
    if (dfn[u] == low[u])
    {
        int y;
        cnt ++ ;
        do
        {
            y = stk[top -- ], ins[y] = false, id[y] = cnt;
        }while (y != u);
    }
}

int main()
{
    scanf("%d", &n);
    memset(h, -1, sizeof h);
    for (int i = 0; i < n; i ++ )
    {
        int s0, s1, t0, t1, d;
        scanf("%d:%d %d:%d %d", &s0, &s1, &t0, &t1, &d);
        w[i] = {s0 * 60 + s1, t0 * 60 + t1, d};
    }

    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < i; j ++ )
        {
            auto a = w[i], b = w[j];
            if (is_overlap(a.s, a.s + a.d, b.s, b.s + b.d)) add(i, j + n), add(j, i + n);
            if (is_overlap(a.s, a.s + a.d, b.t - b.d, b.t)) add(i, j), add(j + n, i + n);
            if (is_overlap(a.t - a.d, a.t, b.s, b.s + b.d)) add(i + n, j + n), add(j, i);
            if (is_overlap(a.t - a.d, a.t, b.t - b.d, b.t)) add(i + n, j), add(j + n, i);
        }

    for (int i = 0; i < n * 2; i ++ )
        if (!dfn[i])
            tarjan(i);

    for (int i = 0; i < n; i ++ )
        if (id[i] == id[i + n])
        {
            puts("NO");
            return 0;
        }

    puts("YES");
    for (int i = 0; i < n; i ++ )
    {
        auto a = w[i];
        int s = a.s, t = a.t, d = a.d;
        if (id[i] < id[i + n])
            printf("%02d:%02d %02d:%02d\n", s / 60, s % 60, (s + d) / 60, (s + d) % 60);
        else
            printf("%02d:%02d %02d:%02d\n", (t - d) / 60, (t - d) % 60, t / 60, t % 60);
    }

    return 0;
}

#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010, M = 200010;

int n, d, m;
char s[N];
int h[N], e[M], ne[M], idx;
int dfn[N], low[N], ts, stk[N], top;
int id[N], cnt;
bool ins[N];
int pos[10];

struct Op
{
    int x, y;
    char a, b;
}op[M];

int get(int x, char b, int t)
{
    char a = s[x] - 'a';
    b -= 'A';
    if (((a + 1) % 3 != b) ^ t) return x + n;
    return x;
}

char put(int x, int t)
{
    int y = s[x] - 'a';
    return 'A' + ((y + t) % 3);
}

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void tarjan(int u)
{
    dfn[u] = low[u] = ++ ts;
    stk[ ++ top] = u, ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u] = min(low[u], low[j]);
        } else if (ins[j]) low[u] = min(low[u], dfn[j]);
    }

    if (dfn[u] == low[u])
    {
        int y;
        cnt ++ ;
        do
        {
            y = stk[top -- ], ins[y] = false, id[y] = cnt;
        } while (y != u);
    }
}


bool work()
{
    memset(h, -1, sizeof h);
    memset(dfn, 0, sizeof dfn);
    idx = ts = cnt = 0;

    for (int i = 0; i < m; i ++ )
    {
        int x = op[i].x - 1, y = op[i].y - 1;
        char a = op[i].a, b = op[i].b;
        if (s[x] != a + 32)
        {
            if (s[y] != b + 32) add(get(x, a, 0), get(y, b, 0)), add(get(y, b, 1), get(x, a, 1));
            else add(get(x, a, 0), get(x, a, 1));
        }
    }

    for (int i = 0; i < n * 2; i ++ )
        if (!dfn[i])
            tarjan(i);

    for (int i = 0; i < n; i ++ )
        if (id[i] == id[i + n])
            return false;

    for (int i = 0; i < n; i ++ )
        if (id[i] < id[i + n]) putchar(put(i, 1));
        else putchar(put(i, 2));

    return true;
}

int main()
{
    scanf("%d%d", &n, &d);
    scanf("%s", s);
    for (int i = 0, j = 0; i < n; i ++ )
        if (s[i] == 'x')
            pos[j ++ ] = i;

    scanf("%d", &m);
    for (int i = 0; i < m; i ++ )
        scanf("%d %c %d %c", &op[i].x, &op[i].a, &op[i].y, &op[i].b);

    for (int k = 0; k < 1 << d; k ++ )
    {
        for (int i = 0; i < d; i ++ )
            if (k >> i & 1) s[pos[i]] = 'a';
            else s[pos[i]] = 'b';

        if (work()) return 0;
    }

    puts("-1");
    return 0;
}

//朱刘算法
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cmath>

#define x first
#define y second

using namespace std;

typedef pair<double, double> PDD;

const int N = 110;
const double INF = 1e8;

int n, m;
PDD q[N];
bool g[N][N];
double d[N][N], bd[N][N];
int pre[N], bpre[N];
int dfn[N], low[N], ts, stk[N], top;
int id[N], cnt;
bool st[N], ins[N];

void dfs(int u)
{
    st[u] = true;
    for (int i = 1; i <= n; i ++ )
        if (g[u][i] && !st[i])
            dfs(i);
}

bool check_con()
{
    memset(st, 0, sizeof st);
    dfs(1);
    for (int i = 1; i <= n; i ++ )
        if (!st[i])
            return false;
    return true;
}

double get_dist(int a, int b)
{
    double dx = q[a].x - q[b].x;
    double dy = q[a].y - q[b].y;
    return sqrt(dx * dx + dy * dy);
}

void tarjan(int u)
{
    dfn[u] = low[u] = ++ ts;
    stk[ ++ top] = u, ins[u] = true;

    int j = pre[u];
    if (!dfn[j])
    {
        tarjan(j);
        low[u] = min(low[u], low[j]);
    } else if (ins[j]) low[u] = min(low[u], dfn[j]);

    if (low[u] == dfn[u])
    {
        int y;
        ++ cnt;
        do
        {
            y = stk[top -- ], ins[y] = false, id[y] = cnt;
        } while (y != u);
    }
}

double work()
{
    double res = 0;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            if (g[i][j]) d[i][j] = get_dist(i, j);
            else d[i][j] = INF;

    while (true)
    {
        for (int i = 1; i <= n; i ++ )
        {
            pre[i] = i;
            for (int j = 1; j <= n; j ++ )
                if (d[pre[i]][i] > d[j][i])
                    pre[i] = j;
        }

        memset(dfn, 0, sizeof dfn);
        ts = cnt = 0;
        for (int i = 1; i <= n; i ++ )
            if (!dfn[i])
                tarjan(i);

        if (cnt == n)
        {
            for (int i = 2; i <= n; i ++ ) res += d[pre[i]][i];
            break;
        }

        for (int i = 2; i <= n; i ++ )
            if (id[pre[i]] == id[i])
                res += d[pre[i]][i];

        for (int i = 1; i <= cnt; i ++ )
            for (int j = 1; j <= cnt; j ++ )
                bd[i][j] = INF;

        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                if (d[i][j] < INF && id[i] != id[j])
                {
                    int a = id[i], b = id[j];
                    if (id[pre[j]] == id[j]) bd[a][b] = min(bd[a][b], d[i][j] - d[pre[j]][j]);
                    else bd[a][b] = min(bd[a][b], d[i][j]);
                }

        n = cnt;
        memcpy(d, bd, sizeof d);
    }

    return res;
}

int main()
{
    while (~scanf("%d%d", &n, &m))
    {
        for (int i = 1; i <= n; i ++ ) scanf("%lf%lf", &q[i].x, &q[i].y);

        memset(g, 0, sizeof g);
        while (m -- )
        {
            int a, b;
            scanf("%d%d", &a, &b);
            if (a != b && b != 1) g[a][b] = true;
        }

        if (!check_con()) puts("poor snoopy");
        else printf("%.2lf\n", work());
    }

    return 0;
}

//Prufer编码
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 100010;

int n, m;
int f[N], d[N], p[N];

void tree2prufer()
{
    for (int i = 1; i < n; i ++ )
    {
        scanf("%d", &f[i]);
        d[f[i]] ++ ;
    }

    for (int i = 0, j = 1; i < n - 2; j ++ )
    {
        while (d[j]) j ++ ;
        p[i ++ ] = f[j];
        while (i < n - 2 && -- d[p[i - 1]] == 0 && p[i - 1] < j) p[i ++ ] = f[p[i - 1]];
    }

    for (int i = 0; i < n - 2; i ++ ) printf("%d ", p[i]);
}

void prufer2tree()
{
    for (int i = 1; i <= n - 2; i ++ )
    {
        scanf("%d", &p[i]);
        d[p[i]] ++ ;
    }
    p[n - 1] = n;

    for (int i = 1, j = 1; i < n; i ++, j ++ )
    {
        while (d[j]) j ++ ;
        f[j] = p[i];
        while (i < n - 1 && -- d[p[i]] == 0 && p[i] < j) f[p[i]] = p[i + 1], i ++ ;
    }

    for (int i = 1; i <= n - 1; i ++ ) printf("%d ", f[i]);
}

int main()
{
    scanf("%d%d", &n, &m);
    if (m == 1) tree2prufer();
    else prufer2tree();

    return 0;
}

#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 210;

int n, m;
int C[N][N], g[N], f[N][N];

void init()
{
    for (int i = 0; i <= n; i ++ )
        for (int j = 0; j <= i; j ++ )
            if (!j) C[i][j] = 1;
            else C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % m;

    g[1] = 1, g[3] = 3;
    for (int i = 4; i <= n; i ++ ) g[i] = g[i - 1] * i % m;
}

int main()
{
    cin >> n >> m;
    init();

    f[0][0] = 1;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= i; j ++ )
            for (int k = 1; k <= i - j + 1; k ++ )
                f[i][j] = (f[i][j] + f[i - k][j - 1] * (LL)C[i - 1][k - 1] * g[k]) % m;

    LL res = g[n - 1], p = 1;
    for (int k = 2; k <= n; k ++ )
    {
        res += f[n][k] * p;
        p = p * n % m;
    }

    cout << res % m << endl;

    return 0;
}

//左偏树
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 1000010;

int n;
int v[N], dist[N], l[N], r[N];
struct Segment
{
    int end, root, size;
}stk[N];
int ans[N];

int merge(int x, int y)
{
    if (!x || !y) return x + y;
    if (v[x] < v[y]) swap(x, y);
    r[x] = merge(r[x], y);
    if (dist[r[x]] > dist[l[x]]) swap(r[x], l[x]);
    dist[x] = dist[r[x]] + 1;
    return x;
}

int pop(int x)
{
    return merge(l[x], r[x]);
}

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )
    {
        scanf("%d", &v[i]);
        v[i] -= i;
    }
    int tt = 0;
    for (int i = 1; i <= n; i ++ )
    {
        auto cur = Segment({i, i, 1});
        dist[i] = 1;
        while (tt && v[cur.root] < v[stk[tt].root])
        {
            cur.root = merge(cur.root, stk[tt].root);
            if (cur.size % 2 && stk[tt].size % 2)
                cur.root = pop(cur.root);
            cur.size += stk[tt].size;
            tt -- ;
        }
        stk[ ++ tt] = cur;
    }

    for (int i = 1, j = 1; i <= tt; i ++ )
    {
        while (j <= stk[i].end)
            ans[j ++ ] = v[stk[i].root];
    }

    LL res = 0;
    for (int i = 1; i <= n; i ++ ) res += abs(v[i] - ans[i]);
    printf("%lld\n", res);
    for (int i = 1; i <= n; i ++ )
        printf("%d ", ans[i] + i);

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 1000010;

int n;
int v[N], dist[N], l[N], r[N];
struct Segment
{
    int end, root, size;
}stk[N];
int ans[N];

int merge(int x, int y)
{
    if (!x || !y) return x + y;
    if (v[x] < v[y]) swap(x, y);
    r[x] = merge(r[x], y);
    if (dist[r[x]] > dist[l[x]]) swap(r[x], l[x]);
    dist[x] = dist[r[x]] + 1;
    return x;
}

int pop(int x)
{
    return merge(l[x], r[x]);
}

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )
    {
        scanf("%d", &v[i]);
        v[i] -= i;
    }
    int tt = 0;
    for (int i = 1; i <= n; i ++ )
    {
        auto cur = Segment({i, i, 1});
        dist[i] = 1;
        while (tt && v[cur.root] < v[stk[tt].root])
        {
            cur.root = merge(cur.root, stk[tt].root);
            if (cur.size % 2 && stk[tt].size % 2)
                cur.root = pop(cur.root);
            cur.size += stk[tt].size;
            tt -- ;
        }
        stk[ ++ tt] = cur;
    }

    for (int i = 1, j = 1; i <= tt; i ++ )
    {
        while (j <= stk[i].end)
            ans[j ++ ] = v[stk[i].root];
    }

    LL res = 0;
    for (int i = 1; i <= n; i ++ ) res += abs(v[i] - ans[i]);
    printf("%lld\n", res);
    for (int i = 1; i <= n; i ++ )
        printf("%d ", ans[i] + i);

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int f[N][11], cost[N][N];
int w[N], v[N], dist[N], l[N], r[N];
struct Segment
{
    int root;
    int tot_sum, tot_size;
    int tree_sum, tree_size;

    int get_cost()
    {
        int mid = v[root];
        return mid * tree_size - tree_sum + tot_sum - tree_sum - (tot_size - tree_size) * mid;
    }
}stk[N];

int merge(int x, int y)
{
    if (!x || !y) return x + y;
    if (v[x] < v[y]) swap(x, y);
    r[x] = merge(r[x], y);
    if (dist[l[x]] < dist[r[x]]) swap(l[x], r[x]);
    dist[x] = dist[r[x]] + 1;
    return x;
}

int pop(int x)
{
    return merge(l[x], r[x]);
}

void get_cost(int u)
{
    int tt = 0, res = 0;
    for (int i = u; i <= n; i ++ )
    {
        auto cur = Segment({i, v[i], 1, v[i], 1});
        l[i] = r[i] = 0, dist[i] = 1;
        while (tt && v[cur.root] < v[stk[tt].root])
        {
            res -= stk[tt].get_cost();
            cur.root = merge(cur.root, stk[tt].root);
            bool is_pop = cur.tot_size % 2 && stk[tt].tot_size % 2;
            cur.tot_size += stk[tt].tot_size;
            cur.tot_sum += stk[tt].tot_sum;
            cur.tree_size += stk[tt].tree_size;
            cur.tree_sum += stk[tt].tree_sum;
            if (is_pop)
            {
                cur.tree_size --;
                cur.tree_sum -= v[cur.root];
                cur.root = pop(cur.root);
            }
            tt -- ;
        }
        stk[ ++ tt] = cur;
        res += cur.get_cost();
        cost[u][i] = min(cost[u][i], res);
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);
    memset(cost, 0x3f, sizeof cost);
    for (int i = 1; i <= n; i ++ ) v[i] = w[i] - i;
    for (int i = 1; i <= n; i ++ ) get_cost(i);
    for (int i = 1; i <= n; i ++ ) v[i] = -w[i] - i;
    for (int i = 1; i <= n; i ++ ) get_cost(i);

    memset(f, 0x3f, sizeof f);
    f[0][0] = 0;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            for (int k = 1; k <= i; k ++ )
                f[i][j] = min(f[i][j], f[i - k][j - 1] + cost[i - k + 1][i]);

    printf("%d\n", f[n][m]);
    return 0;
}

//打表
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1010, M = 1010;

int n;
int w[6][6] = {
    {1, 0, 1, 1, 0, 0},
    {0, 1, 0, 0, 1, 0},
    {0, 1, 0, 0, 1, 0},
    {0, 1, 0, 0, 1, 0},
    {1, 0, 1, 1, 0, 1},
    {0, 0, 0, 0, 1, 0},
};
int f[N][6][M];

void add(int a[], int b[])
{
    for (int i = 0, t = 0; i < M; i ++ )
    {
        t += a[i] + b[i];
        a[i] = t % 10;
        t /= 10;
    }
}

int main()
{
    cin >> n;
    f[1][1][0] = f[1][4][0] = 1;
    for (int i = 2; i < n; i ++ )
        for (int j = 0; j < 6; j ++ )
            for (int k = 0; k < 6; k ++ )
                if (w[k][j])
                    add(f[i][j], f[i - 1][k]);
    int res[M] = {0};
    add(res, f[n - 1][0]), add(res,f[n - 1][4]);
    add(res, res);

    int k = M - 1;
    while (k > 0 && !res[k]) k -- ;
    for (int i = k; i >= 0; i -- ) cout << res[i];
    cout << endl;

    return 0;
}

//manacher算法
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 2e7 + 10;

int n;
char a[N], b[N];
int p[N];

void init()
{
    int k = 0;
    b[k ++ ] = '$', b[k ++ ] = '#';
    for (int i = 0; i < n; i ++ ) b[k ++ ] = a[i], b[k ++ ] = '#';
    b[k ++ ] = '^';
    n = k;
}

void manacher()
{
    int mr = 0, mid;
    for (int i = 1; i < n; i ++ )
    {
        if (i < mr) p[i] = min(p[mid * 2 - i], mr - i);
        else p[i] = 1;
        while (b[i - p[i]] == b[i + p[i]]) p[i] ++ ;
        if (i + p[i] > mr)
        {
            mr = i + p[i];
            mid = i;
        }
    }
}

int main()
{
    scanf("%s", a);
    n = strlen(a);

    init();
    manacher();

    int res = 0;
    for (int i = 0; i < n; i ++ ) res = max(res, p[i]);
    printf("%d\n", res - 1);

    return 0;
}

//最小表示法
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 2000010;

int n;
char a[N], b[N];

int get_min(char s[])
{
    int i = 0, j = 1;
    while (i < n && j < n)
    {
        int k = 0;
        while (k < n && s[i + k] == s[j + k]) k ++ ;
        if (k == n) break;
        if (s[i + k] > s[j + k]) i += k + 1;
        else j += k + 1;
        if (i == j) j ++ ;
    }
    int k = min(i, j);
    s[k + n] = 0;
    return k;
}

int main()
{
    scanf("%s%s", a, b);
    n = strlen(a);
    memcpy(a + n, a, n);
    memcpy(b + n, b, n);

    int x = get_min(a), y = get_min(b);
    if (strcmp(a + x, b + y)) puts("No");
    else
    {
        puts("Yes");
        puts(a + x);
    }

    return 0;
}

//构造
//神奇的幻方
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 40;

int n;
int a[N][N];

int main()
{
    cin >> n;
    int x = 1, y = n / 2 + 1;
    for (int i = 1; i <= n * n; i ++ )
    {
        a[x][y] = i;
        if (x == 1 && y == n) x ++ ;
        else if (x == 1) x = n, y ++ ;
        else if (y == n) x --, y = 1;
        else if (a[x - 1][y + 1]) x ++ ;
        else x --, y ++ ;
    }

    for (int i = 1; i <= n; i ++ )
    {
        for (int j = 1; j <= n; j ++ )
            cout << a[i][j] << ' ';
        cout << endl;
    }
    return 0;
}

//时态同步
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 500010, M = N * 2;

int n, root;
int h[N], e[M], w[M], ne[M], idx;
LL d[N], ans;

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u, int father)
{
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (j == father) continue;
        dfs(j, u);
        d[u] = max(d[u], d[j] + w[i]);
    }

    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (j == father) continue;
        ans += d[u] - (d[j] + w[i]);
    }
}

int main()
{
    scanf("%d%d", &n, &root);
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c), add(b, a, c);
    }

    dfs(root, -1);
    printf("%lld\n", ans);

    return 0;
}

//莫比乌斯反演
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 50010;

int primes[N], cnt, mu[N], sum[N];
bool st[N];

void init()
{
    mu[1] = 1;
    for (int i = 2; i < N; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i, mu[i] = -1;
        for (int j = 0; primes[j] * i < N; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
            mu[primes[j] * i] = -mu[i];
        }
    }
    for (int i = 1; i < N; i ++ ) sum[i] = sum[i - 1] + mu[i];
}

int g(int k, int x)
{
    return k / (k / x);
}

LL f(int a, int b, int k)
{
    a = a / k, b = b / k;
    LL res = 0;
    int n = min(a, b);
    for (int l = 1, r; l <= n; l = r + 1)
    {
        r = min(n, min(g(a, l), g(b, l)));
        res += (LL)(sum[r] - sum[l - 1]) * (a / l) * (b / l);
    }
    return res;
}

int main()
{
    init();
    int T;
    scanf("%d", &T);
    while (T -- )
    {
        int a, b, c, d, k;
        scanf("%d%d%d%d%d", &a, &b, &c, &d, &k);
        printf("%lld\n", f(b, d, k) - f(a - 1, d, k) - f(b, c - 1, k) + f(a - 1, c - 1, k));
    }
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 50010;

int primes[N], cnt, mu[N], sum[N], h[N];
bool st[N];

int g(int k, int x)
{
    return k / (k / x);
}

void init()
{
    mu[1] = 1;
    for (int i = 2; i < N; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i, mu[i] = -1;
        for (int j = 0; primes[j] * i < N; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
            mu[i * primes[j]] = -mu[i];
        }
    }
    for (int i = 1; i < N; i ++ ) sum[i] = sum[i - 1] + mu[i];
    for (int i = 1; i < N; i ++ )
    {
        for (int l = 1, r; l <= i; l = r + 1)
        {
            r = min(i, g(i, l));
            h[i] += (r - l + 1) * (i / l);
        }
    }
}

int main()
{
    init();
    int T;
    scanf("%d", &T);
    while (T -- )
    {
        int n, m;
        scanf("%d%d", &n, &m);
        LL res = 0;
        int k = min(n, m);
        for (int l = 1, r; l <= k; l = r + 1)
        {
            r = min(k, min(g(n, l), g(m, l)));
            res += (LL)(sum[r] - sum[l - 1]) * h[n / l] * h[m / l];
        }
        printf("%lld\n", res);
    }
    return 0;
}

//积性函数
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;

int main()
{
    int n;
    cin >> n;
    LL res = n;
    for (int i = 2; i <= n / i; i ++ )
        if (n % i == 0)
        {
            int a = 0, p = i;
            while (n % p == 0) a ++, n /= p;
            res = res * (p + (LL)a * p - a) / p;
        }
    if (n > 1) res = res * ((LL)n + n - 1) / n;
    cout << res << endl;
    return 0;
}

//BSGS
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <unordered_map>

using namespace std;

typedef long long LL;

int bsgs(int a, int b, int p)
{
    if (1 % p == b % p) return 0;
    int k = sqrt(p) + 1;
    unordered_map<int, int> hash;
    for (int i = 0, j = b % p; i < k; i ++ )
    {
        hash[j] = i;
        j = (LL)j * a % p;
    }
    int ak = 1;
    for (int i = 0; i < k; i ++ ) ak = (LL)ak * a % p;

    for (int i = 1, j = ak; i <= k; i ++ )
    {
        if (hash.count(j)) return (LL)i * k - hash[j];
        j = (LL)j * ak % p;
    }
    return -1;
}

int main()
{
    int a, p, b;
    while (cin >> a >> p >> b, a || p || b)
    {
        int res = bsgs(a, b, p);
        if (res == -1) puts("No Solution");
        else cout << res << endl;
    }
    return 0;
}

//拓展BSGS
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <unordered_map>

using namespace std;

typedef long long LL;
const int INF = 1e8;

int exgcd(int a, int b, int& x, int& y)
{
    if (!b)
    {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int bsgs(int a, int b, int p)
{
    if (1 % p == b % p) return 0;
    int k = sqrt(p) + 1;
    unordered_map<int, int> hash;
    for (int i = 0, j = b % p; i < k; i ++ )
    {
        hash[j] = i;
        j = (LL)j * a % p;
    }
    int ak = 1;
    for (int i = 0; i < k; i ++ ) ak = (LL)ak * a % p;
    for (int i = 1, j = ak; i <= k; i ++ )
    {
        if (hash.count(j)) return i * k - hash[j];
        j = (LL)j * ak % p;
    }
    return -INF;
}

int exbsgs(int a, int b, int p)
{
    b = (b % p + p) % p;
    if (1 % p == b % p) return 0;
    int x, y;
    int d = exgcd(a, p, x, y);
    if (d > 1)
    {
        if (b % d) return -INF;
        exgcd(a / d, p / d, x, y);
        return exbsgs(a, (LL)b / d * x % (p / d), p / d) + 1;
    }
    return bsgs(a, b, p);
}

int main()
{
    int a, p, b;
    while (cin >> a >> p >> b, a || p || b)
    {
        int res = exbsgs(a, b, p);
        if (res < 0) puts("No Solution");
        else cout << res << endl;
    }
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <unordered_map>

using namespace std;

typedef long long LL;

int qmi(int a, int b, int p)
{
    int res = 1;
    while (b)
    {
        if (b & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        b >>= 1;
    }
    return res;
}

int inv(int a, int p)
{
    return qmi(a, p - 2, p);
}

int exgcd(int a, int b, int& x, int& y)
{
    if (!b)
    {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int bsgs(int a, int b, int p)
{
    if (1 % p == b % p) return 0;
    int k = sqrt(p) + 1;
    unordered_map<int, int> hash;
    for (int i = 0, j = b % p; i < k; i ++ )
    {
        hash[j] = i;
        j = (LL)j * a % p;
    }
    int ak = 1;
    for (int i = 0; i < k; i ++ ) ak = (LL)ak * a % p;
    for (int i = 1, j = ak; i <= k; i ++ )
    {
        if (hash.count(j)) return (LL)i * k - hash[j];
        j = (LL)j * ak % p;
    }
    return -2;
}

int main()
{
    int T;
    cin >> T;
    while (T -- )
    {
        int p, a, b, x1, t;
        cin >> p >> a >> b >> x1 >> t;
        if (a == 0)
        {
            if (x1 == t) puts("1");
            else if (b == t) puts("2");
            else puts("-1");
        }
        else if (a == 1)
        {
            if (b == 0) puts(t == x1 ? "1" : "-1");
            else
            {
                int x, y;
                exgcd(b, p, x, y);
                x = ((LL)x * (t - x1) % p + p) % p;
                cout << x + 1 << endl;
            }
        }
        else
        {
            int C = (LL)b * inv(a - 1, p) % p;
            int A = (x1 + C) % p;
            if (A == 0)
            {
                int u = (-C + p) % p;
                puts(u == t ? "1" : "-1");
            }
            else
            {
                int B = (t + C) % p;
                cout << bsgs(a, (LL)B * inv(A, p) % p, p) + 1 << endl;
            }
        }
    }
    return 0;
}

//生成函数
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 510, P = 10007;

char s[N];

int main()
{
    scanf("%s", s);
    LL n = 0;
    for (int i = 0; s[i]; i ++ )
        n = (n * 10 + s[i] - '0') % P;

    cout << n * (n + 1) * (n + 2) / 6 % P << endl;
    return 0;
}

//burnside引理与polya定理
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;

int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}

LL power(int a, int b)
{
    LL res = 1;
    while (b -- ) res *= a;
    return res;
}

int main()
{
    int m, n;
    while (cin >> m >> n, m || n)
    {
        LL res = 0;
        for (int i = 0; i < n; i ++ )
            res += power(m, gcd(n, i));
        if (n % 2)
            res += n * power(m, (n + 1) / 2);
        else
            res += n / 2 * (power(m, n / 2 + 1) + power(m, n / 2));
        cout << res / n / 2 << endl;
    }

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 11, P = 9973;

int m;
struct Matrix
{
    int a[N][N];
    Matrix()
    {
        memset(a, 0, sizeof a);
    }
};

Matrix operator* (Matrix a, Matrix b)
{
    Matrix c;
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= m; j ++ )
            for (int k = 1; k <= m; k ++ )
                c.a[i][j] = (c.a[i][j] + a.a[i][k] * b.a[k][j]) % P;
    return c;
}

int qmi(Matrix a, int b)
{
    Matrix res;
    for (int i = 1; i <= m; i ++ ) res.a[i][i] = 1;
    while (b)
    {
        if (b & 1) res = res * a;
        a = a * a;
        b >>= 1;
    }

    int sum = 0;
    for (int i = 1; i <= m; i ++ ) sum += res.a[i][i];
    return sum % P;
}

int phi(int n)
{
    int res = n;
    for (int i = 2; i * i <= n; i ++ )
        if (n % i == 0)
        {
            res = res / i * (i - 1);
            while (n % i == 0) n /= i;
        }
    if (n > 1) res = res / n * (n - 1);
    return res % P;
}

int inv(int n)
{
    n %= P;
    for (int i = 1; i < P; i ++ )
        if (i * n % P == 1)
            return i;
    return -1;
}

int main()
{
    int T;
    cin >> T;
    while (T -- )
    {
        int n, k;
        cin >> n >> m >> k;
        Matrix tr;
        for (int i = 1; i <= m; i ++ )
            for (int j = 1; j <= m; j ++ )
                tr.a[i][j] = 1;
        while (k -- )
        {
            int x, y;
            cin >> x >> y;
            tr.a[x][y] = tr.a[y][x] = 0;
        }
        int res = 0;
        for (int i = 1; i * i <= n; i ++ )
            if (n % i == 0)
            {
                res = (res + qmi(tr, i) * phi(n / i)) % P;
                if (i != n / i)
                    res = (res + qmi(tr, n / i) * phi(i)) % P;
            }
        cout << res * inv(n) % P << endl;
    }
    return 0;
}

//斯特林数
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 1010, MOD = 1e9 + 7;

int n, m;
int f[N][N];

int main()
{
    cin >> n >> m;
    f[0][0] = 1;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            f[i][j] = (f[i - 1][j - 1] + (LL)(i - 1) * f[i - 1][j]) % MOD;
    cout << f[n][m] << endl;
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 1010, MOD = 1e9 + 7;

int n, m;
int f[N][N];

int main()
{
    cin >> n >> m;
    f[0][0] = 1;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            f[i][j] = (f[i - 1][j - 1] + (LL)j * f[i - 1][j]) % MOD;
    cout << f[n][m] << endl;
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 50010, M = 210, MOD = 1e9 + 7;

int f[N][M], c[M][M];

int main()
{
    f[0][0] = 1;
    for (int i = 1; i < N; i ++ )
        for (int j = 1; j < M; j ++ )
            f[i][j] = (f[i - 1][j - 1] + (LL)(i - 1) * f[i - 1][j]) % MOD;
    for (int i = 0; i < M; i ++ )
        for (int j = 0; j <= i; j ++ )
            if (!j) c[i][j] = 1;
            else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % MOD;

    int T;
    scanf("%d", &T);
    while (T -- )
    {
        int n, a, b;
        scanf("%d%d%d", &n, &a, &b);
        printf("%lld\n", (LL)f[n - 1][a + b - 2] * c[a + b - 2][a - 1] % MOD);
    }

    return 0;
}

//线性基
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 100010;

int n;
LL a[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%lld", &a[i]);

    int k = 0;
    for (int i = 62; i >= 0; i -- )
    {
        for (int j = k; j < n; j ++ )
            if (a[j] >> i & 1)
            {
                swap(a[j], a[k]);
                break;
            }
        if (!(a[k] >> i & 1)) continue;
        for (int j = 0; j < n; j ++ )
            if (j != k && (a[j] >> i & 1))
                a[j] ^= a[k];
        k ++ ;
        if (k == n) break;
    }

    LL res = 0;
    for (int i = 0; i < k; i ++ ) res ^= a[i];
    printf("%lld\n", res);
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 10010;

LL a[N];

int main()
{
    int T;
    scanf("%d", &T);
    for (int C = 1; C <= T; C ++ )
    {
        printf("Case #%d:\n", C);
        int n;
        scanf("%d", &n);
        for (int i = 0; i < n; i ++ ) scanf("%lld", &a[i]);
        int k = 0;
        for (int i = 62; i >= 0; i -- )
        {
            for (int j = k; j < n; j ++ )
                if (a[j] >> i & 1)
                {
                    swap(a[j], a[k]);
                    break;
                }
            if (!(a[k] >> i & 1)) continue;
            for (int j = 0; j < n; j ++ )
                if (j != k && (a[j] >> i & 1))
                    a[j] ^= a[k];
            k ++ ;
            if (k == n) break;
        }
        reverse(a, a + k);

        int m;
        scanf("%d", &m);
        while (m -- )
        {
            LL x;
            scanf("%lld", &x);
            if (k < n) x -- ;
            if (x >= (1ll << k)) puts("-1");
            else
            {
                LL res = 0;
                for (int i = 0; i < k; i ++ )
                    if (x >> i & 1)
                        res ^= a[i];
                printf("%lld\n", res);
            }
        }
    }
    return 0;
}


//FFT
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace std;

const int N = 300010;
const double PI = acos(-1);

int n, m;
struct Complex
{
    double x, y;
    Complex operator+ (const Complex& t) const
    {
        return {x + t.x, y + t.y};
    }
    Complex operator- (const Complex& t) const
    {
        return {x - t.x, y - t.y};
    }
    Complex operator* (const Complex& t) const
    {
        return {x * t.x - y * t.y, x * t.y + y * t.x};
    }
}a[N], b[N];
int rev[N], bit, tot;

void fft(Complex a[], int inv)
{
    for (int i = 0; i < tot; i ++ )
        if (i < rev[i])
            swap(a[i], a[rev[i]]);
    for (int mid = 1; mid < tot; mid <<= 1)
    {
        auto w1 = Complex({cos(PI / mid), inv * sin(PI / mid)});
        for (int i = 0; i < tot; i += mid * 2)
        {
            auto wk = Complex({1, 0});
            for (int j = 0; j < mid; j ++, wk = wk * w1)
            {
                auto x = a[i + j], y = wk * a[i + j + mid];
                a[i + j] = x + y, a[i + j + mid] = x - y;
            }
        }
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 0; i <= n; i ++ ) scanf("%lf", &a[i].x);
    for (int i = 0; i <= m; i ++ ) scanf("%lf", &b[i].x);
    while ((1 << bit) < n + m + 1) bit ++;
    tot = 1 << bit;
    for (int i = 0; i < tot; i ++ )
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    fft(a, 1), fft(b, 1);
    for (int i = 0; i < tot; i ++ ) a[i] = a[i] * b[i];
    fft(a, -1);
    for (int i = 0; i <= n + m; i ++ )
        printf("%d ", (int)(a[i].x / tot + 0.5));

    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace std;

const int N = 300000;
const double PI = acos(-1);

struct Complex
{
    double x, y;
    Complex operator+ (const Complex& t) const
    {
        return {x + t.x, y + t.y};
    }
    Complex operator- (const Complex& t) const
    {
        return {x - t.x, y - t.y};
    }
    Complex operator* (const Complex& t) const
    {
        return {x * t.x - y * t.y, x * t.y + y * t.x};
    }
}a[N], b[N];
char s1[N], s2[N];
int res[N];
int rev[N], bit, tot;

void fft(Complex a[], int inv)
{
    for (int i = 0; i < tot; i ++ )
        if (i < rev[i])
            swap(a[i], a[rev[i]]);
    for (int mid = 1; mid < tot; mid *= 2)
    {
        auto w1 = Complex({cos(PI / mid), inv * sin(PI / mid)});
        for (int i = 0; i < tot; i += mid * 2)
        {
            auto wk = Complex({1, 0});
            for (int j = 0; j < mid; j ++, wk = wk * w1)
            {
                auto x = a[i + j], y = wk * a[i + j + mid];
                a[i + j] = x + y, a[i + j + mid] = x - y;
            }
        }
    }
}

int main()
{
    scanf("%s%s", s1, s2);
    int n = strlen(s1) - 1, m = strlen(s2) - 1;
    for (int i = 0; i <= n; i ++ ) a[i].x = s1[n - i] - '0';
    for (int i = 0; i <= m; i ++ ) b[i].x = s2[m - i] - '0';
    while ((1 << bit) < n + m + 1) bit ++ ;
    tot = 1 << bit;
    for (int i = 0; i < tot; i ++ )
        rev[i] = ((rev[i >> 1] >> 1)) | ((i & 1) << (bit - 1));
    fft(a, 1), fft(b, 1);
    for (int i = 0; i < tot; i ++ ) a[i] = a[i] * b[i];
    fft(a, -1);

    int k = 0;
    for (int i = 0, t = 0; i < tot || t; i ++ )
    {
        t += a[i].x / tot + 0.5;
        res[k ++ ] = t % 10;
        t /= 10;
    }

    while (k > 1 && !res[k - 1]) k -- ;
    for (int i = k - 1; i >= 0; i -- ) printf("%d", res[i]);

    return 0;
}

//基环树DP
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 1200010, M = N * 2;

int n;
int h[N], e[M], w[M], ne[M], idx;
int fu[N], fw[N], q[N];
int cir[N], ed[N], cnt;
LL s[N], d[N * 2], sum[N * 2];
bool st[N], ins[N];
LL ans;

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs_c(int u, int from)
{
    st[u] = ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
    {
        if (i == (from ^ 1)) continue;
        int j = e[i];
        fu[j] = u, fw[j] = w[i];
        if (!st[j]) dfs_c(j, i);
        else if (ins[j])
        {
            cnt ++ ;
            ed[cnt] = ed[cnt - 1];
            LL sum = w[i];
            for (int k = u; k != j; k = fu[k])
            {
                s[k] = sum;
                sum += fw[k];
                cir[ ++ ed[cnt]] = k;
            }
            s[j] = sum, cir[ ++ ed[cnt]] = j;
        }
    }

    ins[u] = false;
}

LL dfs_d(int u)
{
    st[u] = true;
    LL d1 = 0, d2 = 0;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (st[j]) continue;
        LL dist = dfs_d(j) + w[i];
        if (dist >= d1) d2 = d1, d1 = dist;
        else if (dist > d2) d2 = dist;
    }
    ans = max(ans, d1 + d2);
    return d1;
}

int main()
{
    scanf("%d", &n);
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(i, a, b), add(a, i, b);
    }
    for (int i = 1; i <= n; i ++ )
        if (!st[i])
            dfs_c(i, -1);

    memset(st, 0, sizeof st);
    for (int i = 1; i <= ed[cnt]; i ++ ) st[cir[i]] = true;

    LL res = 0;
    for (int i = 1; i <= cnt; i ++ )
    {
        ans = 0;
        int sz = 0;
        for (int j = ed[i - 1] + 1; j <= ed[i]; j ++ )
        {
            int k = cir[j];
            d[sz] = dfs_d(k);
            sum[sz] = s[k];
            sz ++ ;
        }
        for (int j = 0; j < sz; j ++ )
            d[sz + j] = d[j], sum[sz + j] = sum[j] + sum[sz - 1];
        int hh = 0, tt = -1;
        for (int j = 0; j < sz * 2; j ++ )
        {
            if (hh <= tt && j - q[hh] >= sz) hh ++ ;
            if (hh <= tt) ans = max(ans, d[j] + sum[j] + d[q[hh]] - sum[q[hh]]);
            while (hh <= tt && d[q[tt]] - sum[q[tt]] <= d[j] - sum[j]) tt -- ;
            q[ ++ tt] = j;
        }
        res += ans;
    }

    printf("%lld\n", res);
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 1000010, INF = 1e9;

int n;
int h[N], e[N], rm[N], w[N], ne[N], idx;
LL f1[N][2], f2[N][2];
bool st[N], ins[N];
LL ans;

inline void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs_f(int u, int ap, LL f[][2])
{
    for (int i = h[u]; ~i; i = ne[i])
    {
        if (rm[i]) continue;
        int j = e[i];
        dfs_f(j, ap, f);
        f[u][0] += max(f[j][0], f[j][1]);
    }

    f[u][1] = -INF;
    if (u != ap)
    {
        f[u][1] = w[u];
        for (int i = h[u]; ~i; i = ne[i])
        {
            if (rm[i]) continue;
            int j = e[i];
            f[u][1] += f[j][0];
        }
    }
}

void dfs_c(int u, int from)
{
    st[u] = ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!st[j]) dfs_c(j, i);
        else if (ins[j])
        {
            rm[i] = 1;
            dfs_f(j, -1, f1);
            dfs_f(j, u, f2);
            ans += max(f1[j][0], f2[j][1]);
        }
    }

    ins[u] = false;
}

int main()
{
    scanf("%d", &n);
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(b, i);
        w[i] = a;
    }
    for (int i = 1; i <= n; i ++ )
        if (!st[i])
            dfs_c(i, -1);

    printf("%lld\n", ans);
    return 0;
}

#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1000010, INF = 1e8;

int n;
int h[N], e[N], rm[N], ne[N], idx;
int f1[N][2], f2[N][2];
bool st[N], ins[N];
int ans;

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs_f(int u, int ap, int f[][2])
{
    for (int i = h[u]; ~i; i = ne[i])
    {
        if (rm[i]) continue;
        int j = e[i];
        dfs_f(j, ap, f);
        f[u][0] += max(f[j][0], f[j][1]);
    }
    if (u == ap) f[u][1] = f[u][0] + 1, f[u][0] = -INF;
    else
    {
        f[u][1] = -INF;
        for (int i = h[u]; ~i; i = ne[i])
        {
            if (rm[i]) continue;
            int j = e[i];
            f[u][1] = max(f[u][1], f[u][0] - max(f[j][0], f[j][1]) + f[j][0] + 1);
        }
    }
}

void dfs_c(int u, int from)
{
    st[u] = ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!st[j]) dfs_c(j, i);
        else if (ins[j])
        {
            rm[i] = 1;
            dfs_f(j, -1, f1);
            dfs_f(j, u, f2);
            ans += max(max(f1[j][0], f1[j][1]), f2[j][0]);
        }
    }

    ins[u] = false;
}

int main()
{
    scanf("%d", &n);
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++ )
    {
        int a;
        scanf("%d", &a);
        add(a, i);
    }

    for (int i = 1; i <= n; i ++ )
        if (!st[i])
            dfs_c(i, -1);

    printf("%d\n", ans);
    return 0;
}

//插头DP
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;
const int N = 50000, M = N * 2 + 7;

int n, m, end_x, end_y;
int g[20][20], q[2][N], cnt[N];
int h[2][M];
LL v[2][M];

int find(int cur, int x)
{
    int t = x % M;
    while (h[cur][t] != -1 && h[cur][t] != x)
        if ( ++ t == M)
            t = 0;
    return t;
}

void insert(int cur, int state, LL w)
{
    int t = find(cur, state);
    if (h[cur][t] == -1)
    {
        h[cur][t] = state, v[cur][t] = w;
        q[cur][ ++ cnt[cur]] = t;
    }
    else v[cur][t] += w;
}

int get(int state, int k)  // 求第k个格子的状态，四进制的第k位数字
{
    return state >> k * 2 & 3;
}

int set(int k, int v)  // 构造四进制的第k位数字为v的数
{
    return v * (1 << k * 2);
}

int main()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i ++ )
    {
        char str[20];
        scanf("%s", str + 1);
        for (int j = 1; j <= m; j ++ )
            if (str[j] == '.')
            {
                g[i][j] = 1;
                end_x = i, end_y = j;
            }
    }

    LL res = 0;
    memset(h, -1, sizeof h);
    int cur = 0;
    insert(cur, 0, 1);
    for (int i = 1; i <= n; i ++ )
    {
        for (int j = 1; j <= cnt[cur]; j ++ )
            h[cur][q[cur][j]] <<= 2;
        for (int j = 1; j <= m; j ++ )
        {
            int last = cur;
            cur ^= 1, cnt[cur] = 0;
            memset(h[cur], -1, sizeof h[cur]);
            for (int k = 1; k <= cnt[last]; k ++ )
            {
                int state = h[last][q[last][k]];
                LL w = v[last][q[last][k]];
                int x = get(state, j - 1), y = get(state, j);
                if (!g[i][j])
                {
                    if (!x && !y) insert(cur, state, w);
                }
                else if (!x && !y)
                {
                    if (g[i + 1][j] && g[i][j + 1])
                        insert(cur, state + set(j - 1, 1) + set(j, 2), w);
                }
                else if (!x && y)
                {
                    if (g[i][j + 1]) insert(cur, state, w);
                    if (g[i + 1][j]) insert(cur, state + set(j - 1, y) - set(j, y), w);
                }
                else if (x && !y)
                {
                    if (g[i][j + 1]) insert(cur, state - set(j - 1, x) + set(j, x), w);
                    if (g[i + 1][j]) insert(cur, state, w);
                }
                else if (x == 1 && y == 1)
                {
                    for (int u = j + 1, s = 1;; u ++ )
                    {
                        int z = get(state, u);
                        if (z == 1) s ++ ;
                        else if (z == 2)
                        {
                            if ( -- s == 0)
                            {
                                insert(cur, state - set(j - 1, x) - set(j, y) - set(u, 1), w);
                                break;
                            }
                        }
                    }
                }
                else if (x == 2 && y == 2)
                {
                    for (int u = j - 2, s = 1;; u -- )
                    {
                        int z = get(state, u);
                        if (z == 2) s ++ ;
                        else if (z == 1)
                        {
                            if ( -- s == 0)
                            {
                                insert(cur, state - set(j - 1, x) - set(j, y) + set(u, 1), w);
                                break;
                            }
                        }
                    }
                }
                else if (x == 2 && y == 1)
                {
                    insert(cur, state - set(j - 1, x) - set(j, y), w);
                }
                else if (i == end_x && j == end_y)
                    res += w;
            }
        }
    }

    cout << res << endl;

    return 0 ;
}