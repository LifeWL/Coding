#### [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

 

**示例:**

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

 

```C++
class MinStack {
public:
    stack<int> stk1;
    stack<int> stk2;
    MinStack() {
        stk2.push(INT_MAX);
    }
    
    void push(int x) {
        stk1.push(x);
        stk2.push(std::min(stk2.top(), x));
    }
    
    void pop() {
        stk2.pop();
        stk1.pop();
    }
    
    int top() {
        return stk1.top();
    }
    
    int min() {
        return stk2.top();
    }
};
```

