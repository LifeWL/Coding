#### [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

难度简单424收藏分享切换为英文接收动态反馈

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

 

**示例 1：**

```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

**示例 2：**

```
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

 

```C++
class Solution {
public:
    vector<int> res;
    int partition(vector<int>& arr, int l, int r){
        int t = arr[l];
        int i = l, j = r + 1;
        while(true){
            while(++i <= r && arr[i] < t);
            while(--j >= l && arr[j] > t);
            if(i >= j){
                break;
            }
            swap(arr[i], arr[j]);
        }
        swap(arr[l], arr[j]);
        return j;
    }

    vector<int> quickSelection(vector<int>& arr, int l, int r, int k){
        int mid = partition(arr, l, r);
        if(mid == k){
            res.assign(arr.begin(), arr.begin() + k + 1);
            return res;
        }
        return mid < k ? quickSelection(arr, mid + 1, r, k) : quickSelection(arr, l, mid - 1, k);
    }

    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        if(arr.size() == 0 || k == 0){
            return res;
        }
        return quickSelection(arr, 0, arr.size() - 1, k - 1);
    }
};
```

