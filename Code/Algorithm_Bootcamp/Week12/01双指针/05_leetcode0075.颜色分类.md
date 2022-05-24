#### [75. 颜色分类](https://leetcode.cn/problems/sort-colors/)

给定一个包含红色、白色和蓝色、共 `n` 个元素的数组 `nums` ，**[原地](https://baike.baidu.com/item/原地算法)**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 `0`、 `1` 和 `2` 分别表示红色、白色和蓝色。



必须在不使用库的sort函数的情况下解决这个问题。

 

**示例 1：**

```
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

**示例 2：**

```
输入：nums = [2,0,1]
输出：[0,1,2]
```

 

```C++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        // 交换2和⾮2, 把2甩到最右侧
        int p = 0;
        int q = nums.size() - 1;
        while (p < q) {
            if (nums[p] != 2) {
                p++;
                continue;
            }
            if (nums[q] == 2) {
                q--;
                continue;
            }
            swap(nums[p++], nums[q--]);
        }
        // 交换0和1, 把0甩到最左
        int i = 0;
        int j = p;
        if (nums[j] == 2) j--;
        while (i < j) {
            if (nums[i] == 0) {
                i++;
                continue;
            }
            if (nums[j] == 1) {
                j--;
                continue;
            }
            swap(nums[i++], nums[j--]);
        }
    }
};
```

