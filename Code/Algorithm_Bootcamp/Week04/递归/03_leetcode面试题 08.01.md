#### 面试题 08.01. 三步问题

三步问题。有个小孩正在上楼梯，楼梯有n阶台阶，小孩一次可以上1阶、2阶或3阶。实现一种方法，计算小孩有多少种上楼梯的方式。结果可能很大，你需要对结果模1000000007。

**示例1:**

```
 输入：n = 3 
 输出：4
 说明: 有四种走法
```

**示例2:**

```
 输入：n = 5
 输出：13
```

**C++**

```c++
```

**Java**

```java
class Solution { 
    private int mod = 1000000007; 
    private int[] memo = new int[1000001]; 
    public int waysToStep(int n) { 
        if (n == 1) return 1; 
        if (n == 2) return 2; 
        if (n == 3) return 4; 
        if (memo[n] != 0) return memo[n]; 
        memo[n] = ((waysToStep(n - 1) + waysToStep(n - 2)) % mod + waysToStep(n - 3)) % mod; 
        return memo[n]; 
    } 
}
```

