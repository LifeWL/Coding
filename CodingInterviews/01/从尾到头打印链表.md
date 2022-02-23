#### [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

**示例 1：**

```
输入：head = [1,3,2]
输出：[2,3,1]
```



```c++
使用栈
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
    vector<int> reversePrint(ListNode* head) {
        stack<int> stk;
        vector<int> vec;
        while (head != NULL) {
            stk.push(head->val);
            head = head->next;
        }
        while(!stk.empty()) {
            vec.push_back(stk.top()) ;
            stk.pop();
        }
        return vec;
    }
};
```



```c++
翻转
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
    vector<int> reversePrint(ListNode* head) {
        vector<int> vec;
        while(head)
        {
            vec.push_back(head->val);
            head = head->next;
        }
        reverse(vec.begin(), vec.end());
        return vec;
    }
};
```



```c++
递归
 /**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
    vector<int> vec;
public:
    vector<int>& reversePrint(ListNode* head) { 
        if(head == NULL)    return vec;
        reversePrint(head->next);
        vec.push_back(head->val);
        return vec;
    }
};
```

