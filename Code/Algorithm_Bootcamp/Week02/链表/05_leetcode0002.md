#### 2. 两数相加]

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/02/addtwonumber1.jpg)

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

**示例 2：**

```
输入：l1 = [0], l2 = [0]
输出：[0]
```

**示例 3：**

```
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

 

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* head =new ListNode(0);
        ListNode* temp = head;
        ListNode* temp1 = l1;
        ListNode* temp2 = l2;
        int length1 = 0;
        int length2 = 0;
        while (temp1->next != nullptr) {
            length1++;
            temp1 = temp1->next;
        }
        while (temp2->next != nullptr) {
            length2++;
            temp2 = temp2->next;
        }
        if (length1 > length2) {
            for (int i = 0; i < length1 - length2; i++) {
                temp2->next = new ListNode(0);
                temp2 = temp2->next;
            }
        } else {
            for (int i = 0; i < length2 - length1; i++) {
                temp1->next = new ListNode(0);
                temp1 = temp1->next;
            }
        }
        bool count=false;
        int i = 0;
        while (l1 && l2) {       
            i = count +l1->val + l2->val;
            temp->next =new ListNode(i % 10);
            count = i >= 10 ? true : false;
            temp = temp->next;
            l1 = l1->next;
            l2 = l2->next;
        }
        if(count) {
            temp->next = new ListNode(1);
        }
        return head->next;
    }
};
```

