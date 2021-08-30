# 设计一个有getMin功能的栈

#### 描述

实现一个特殊功能的栈，在实现栈的基本功能的基础上，再实现返回栈中最小元素的操作。

#### 输入描述：

第一行输入一个整数N，表示对栈进行的操作总数。

下面N行每行输入一个字符串S，表示操作的种类。

如果S为"push"，则后面还有一个整数X表示向栈里压入整数X。

如果S为"pop"，则表示弹出栈顶操作。

如果S为"getMin"，则表示询问当前栈中的最小元素是多少。

#### 输出描述：

对于每个getMin操作，输出一行表示当前栈中的最小元素是多少。

#### 示例1

```
输入：
    6
    push 3
    push 2
    push 1
    getMin
    pop
    getMin
输出：
    1
    2
```



#### 第一种实现

使用两个栈， 一个主栈正常操作， 一个getMin栈放最小值， 当主栈压入值比getMin的栈顶小时，  更新getMin的栈顶。

```c++
#include <iostream>
#include <stack>
#include <string>

using namespace std;

class Solution{
  private:
  stack<int> stackData;
  stack<int> stackMin;
  
  public:
  void push(int newNum) {
    if (this->stackData.empty()){
      this->stackMin.push(newNum);
    } else if (newNum < this->getMin()) {
      this->stackMin.push(newNum);
    } else {
      int newMin = this->stackMin.top();
      this->stackMin.push(newMin);
    }
    this->stackData.push(newNum);
  }
  
  int getMin() {
   if (this->stackMin.empty()) {
     return 0;
   }
    return this->stackMin.top();
  }
  
  void pop() {
    if (this->stackData.empty()) {
      return ;
    }
    this->stackMin.pop();
    return this->stackData.pop();
  }
};

static Solution obj;

int main()
{
  int n, value;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s == "push")
    {
      cin >> value;
      obj.push(value);
    }
    else if (s == "getMin") {
      cout << obj.getMin() << endl;
    }
    else {
      obj.pop();
    }
  }
  return 0;
}
```
#### 第二种实现

也是使用两个栈 ，与第一种的区别是主栈更新时， getMin也同步更新， 即如果主栈压入值比getMin的栈顶小时，两栈都压入这个值， 如果更大时，getMin再压一遍栈顶元素。 

```c++
#include <iostream>
#include <stack>
#include <string>

using namespace std;

class Solution{
  private:
  stack<int> stackData;
  stack<int> stackMin;
  
  public:
  void push(int newNum) {
    if (this->stackData.empty()){
      this->stackMin.push(newNum);
    } else if (newNum <= this->getMin()) {
      this->stackMin.push(newNum);
    } 
    this->stackData.push(newNum);
  }
  
  int getMin() {
   if (this->stackMin.empty()) {
     return 0;
   }
    return this->stackMin.top();
  }
  
  void pop() {
    if (this->stackData.empty()) {
      return ;
    }
    int value = this->stackData.top();
    if (value == this->getMin()) {
      this->stackMin.pop();
    }
    return this->stackData.pop();
  }
};

static Solution obj;

int main()
{
  int n, value;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s == "push")
    {
      cin >> value;
      obj.push(value);
    }
    else if (s == "getMin") {
      cout << obj.getMin() << endl;
    }
    else {
      obj.pop();
    }
  }
  return 0;
}
```

