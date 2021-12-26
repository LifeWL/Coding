#### 剑指 Offer 06. 从尾到头打印链表

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

 

**示例 1：**

```
输入：head = [1,3,2]
输出：[2,3,1]
```

 



**C++**

```c++
```



**Java**

```java
class Solution {
    public int[] reversePrint(ListNode head) {
        if (head == null) return new int[0];
        int[] subresult = reversePrint(head.next);
        int[] result = new int[subresult.length+1];
        for (int i = 0; i < subresult.length; ++i) {
            result[i] = subresult[i];
        }
        result[result.length-1] = head.val;
        return result;
    } 
}
```

