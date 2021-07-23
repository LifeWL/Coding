# 设计一个有getMin功能的栈

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

