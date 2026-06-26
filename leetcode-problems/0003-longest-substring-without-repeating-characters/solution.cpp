class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> res;
        int n = s.length();
        int left = 0;
        int maxlen = 0;
        for(int right = 0; right< n; right++){
            if(res.count(s[right]) && res[s[right]] >= left){
                left = res[s[right]] + 1;
            }
            res[s[right]] =  right;
            maxlen = max(maxlen, right - left + 1);
            
        }
        return maxlen;
    }
};
