#### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

**示例 1：**

```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

**示例 2：**

```
输入：lists = []
输出：[]
```

**示例 3：**

```
输入：lists = [[]]
输出：[]
```

 

```C++
class Solution {
public:
    struct cmp {
        bool operator() (ListNode* node1, ListNode* node2) {
            return node1->val > node2->val; // ⼩顶堆
        }
    };
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        priority_queue<ListNode*, vector<ListNode*>, cmp> minQ;
        for (int i = 0; i < lists.size(); ++i) {
            if (lists[i]) {
            minQ.push(lists[i]);
        }      
    }
    ListNode* dummyNode = new ListNode(0);
    ListNode* tail = dummyNode;
    while (!minQ.empty()) {
        ListNode* curNode = minQ.top();
        minQ.pop();
        tail->next = curNode;
        tail = curNode;
        if (curNode->next) {
            minQ.push(curNode->next);
        }
    }
    return dummyNode->next;
    }
};
```

