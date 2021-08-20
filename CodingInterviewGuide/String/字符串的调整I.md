# **字符串的调整I**

```C++
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

void modify(string& str) {
    if (str.empty()) {
        return;
    }
    
    int j = str.size() - 1;
    for (int i = str.size() - 1; i >= 0; --i) {
        if (str[i] != '*') {
            str[j--] = str[i];
        }
    }
    
    for (; j >= 0; --j) {
        str[j] = '*';
    }
}

int main() {
    string str;
    cin >> str;
    
    modify(str);
    cout << str << endl;
    
    return 0;
}
```

