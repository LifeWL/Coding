#### 148. 排序链表

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

**进阶：**

- 你可以在 `O(n log n)` 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/14/sort_list_1.jpg)

```
输入：head = [4,2,1,3]
输出：[1,2,3,4]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/09/14/sort_list_2.jpg)

```
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
```

**示例 3：**

```
输入：head = []
输出：[]
```

**C++**

```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;

        for (int i = 1; i < n; i *= 2) {
            auto dummy = new ListNode(-1), cur = dummy;
            for (int j = 1; j <= n; j += i * 2) {
                auto p = head, q = p;
                for (int k = 0; k < i && q; k ++ ) q = q->next;
                auto o = q;
                for (int k = 0; k < i && o; k ++ ) o = o->next;
                int l = 0, r = 0;
                while (l < i && r < i && p && q)
                    if (p->val <= q->val) cur = cur->next = p, p = p->next, l ++ ;
                    else cur = cur->next = q, q = q->next, r ++ ;
                while (l < i && p) cur = cur->next = p, p = p->next, l ++ ;
                while (r < i && q) cur = cur->next = q, q = q->next, r ++ ;
                head = o;
            }
            cur->next = NULL;
            head = dummy->next;
        }

        return head;
    }
};
```
