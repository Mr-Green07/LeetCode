class Solution {
public:
    string smallestPalindrome(string s, int k) {
        vector<int> count(26, 0);
        for (char c : s) count[c - 'a']++;
        
        vector<int> half(26, 0);
        char mid = 0;
        for (int i = 0; i < 26; i++) {
            half[i] = count[i] / 2;
            if (count[i] % 2 == 1) mid = 'a' + i;
        }
        
        if (countArrangements(half) < k) return "";
        
        int halfLen = 0;
        for (int i = 0; i < 26; i++) halfLen += half[i];
        
        string left;
        for (int pos = 0; pos < halfLen; pos++) {
            for (int i = 0; i < 26; i++) {
                if (half[i] == 0) continue;
                half[i]--;
                long long arr = countArrangements(half);
                if (arr >= k) {
                    left += ('a' + i);
                    break;
                }
                k -= arr;
                half[i]++;
            }
        }
        
        string result = left;
        if (mid) result += mid;
        reverse(left.begin(), left.end());
        result += left;
        return result;
    }
    
private:
    static const int MAX = 1000001;
    
    long long countArrangements(const vector<int>& cnt) {
        int total = 0;
        for (int i = 0; i < 26; i++) total += cnt[i];
        long long res = 1;
        for (int i = 0; i < 26; i++) {
            if (cnt[i] == 0) continue;
            res *= nCk(total, cnt[i]);
            if (res >= MAX) return MAX;
            total -= cnt[i];
        }
        return res;
    }
    
    long long nCk(int n, int k) {
        long long res = 1;
        for (int i = 1; i <= min(k, n - k); i++) {
            res = res * (n - i + 1) / i;
            if (res >= MAX) return MAX;
        }
        return res;
    }
};
