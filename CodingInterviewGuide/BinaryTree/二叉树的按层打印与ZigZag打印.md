# **二叉树的按层打印与ZigZag打印**

```C++
#include<iostream>
#include <queue>
#include <algorithm>
using namespace std;
#define random(a,b) (rand()%(b-a)+a)

struct Node {
    int left;
    int right;
}a[500000];

void bfs(int root){
    
    queue<int> queues;
    queues.push(root);
    
    vector<vector<int>> t;
    vector<int> b;
    
    int height = 0;
    
    while(!queues.empty())
    {
        int size = queues.size();
        height++;
        cout<<"Level "<<height<<" : ";
        for(int i = 0; i < size; i++)
        {
            int cur = queues.front();
            cout<<cur<<" ";
            b.push_back(cur);
            queues.pop();
            if(a[cur].left != 0) {
                queues.push(a[cur].left);
            }
            if(a[cur].right != 0) {
                queues.push(a[cur].right);
            }
        }
        if(height % 2 ==0)
        {
            reverse(b.begin(), b.end());
        }
        t.push_back(b);
        b.clear();
        cout<<endl;
    }
    for(int i = 0; i < t.size(); i ++)
    {
        if(i % 2 == 0)
        {
            cout<<"Level "<<i+1<<" from left to right: ";
        }
        else
        {
            cout<<"Level "<<i+1<<" from right to left: ";
        }
        for(int j = 0; j < t[i].size(); j++)
        {
            cout<<t[i][j]<<" ";
        }
        cout<<endl;
    }
}

int main()
{
    int n,root;
    scanf("%d%d",&n,&root);
    for(int i=1;i<=n;i++)
    {
        int fa,lch,rch;
        scanf("%d%d%d",&fa,&lch,&rch);
        a[fa].left=lch;
        a[fa].right=rch;
    }
    bfs(root);
    
}
```

