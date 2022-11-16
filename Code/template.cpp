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

vector<int> add(vector<int> &A, vector<int> &B)
{
    if (A.size() < B.size()) return add(B, A);
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
