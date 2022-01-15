#### [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

**[Trie](https://baike.baidu.com/item/字典树/9825209?fr=aladdin)**（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

- `Trie()` 初始化前缀树对象。
- `void insert(String word)` 向前缀树中插入字符串 `word` 。
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`（即，在检索之前已经插入）；否则，返回 `false` 。
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix` ，返回 `true` ；否则，返回 `false` 。

 

**示例：**

```
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

 

```C++
class Trie {
public:
    class TrieNode {
    public:
        char data;
        bool isEnding = false;
        vector<TrieNode*> children;
        TrieNode(char data) {
            this->data = data;
            children.assign(26, nullptr);
        }
    };
    TrieNode* root;
    Trie() {
        root = new TrieNode('/');
    }

    void insert(string word) {
        TrieNode* p = root;
        for (int i = 0; i < word.size(); ++i) {
            char c = word[i];
            if (p->children[c - 'a'] == nullptr) {
                p->children[c - 'a'] = new TrieNode(c);
            }
            p = p->children[c - 'a'];
        }
        p->isEnding = true;
    }

    bool search(string word) {
        TrieNode* p = root;
        for (int i = 0; i < word.size(); ++i) {
           char c = word[i];
           if (p->children[c - 'a'] == nullptr) return false;
           p = p->children[c - 'a'];
           }
        return p->isEnding;
    }
    
    bool startsWith(string prefix) {
       TrieNode* p = root;
       for (int i = 0; i < prefix.size(); ++i) {
       char c = prefix[i];
       if (p->children[c - 'a'] == nullptr) return false;
           p = p->children[c - 'a'];
       }
       return true;
    }
};
```

