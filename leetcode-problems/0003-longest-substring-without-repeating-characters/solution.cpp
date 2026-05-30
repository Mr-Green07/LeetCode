class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> ans;
        int left = 0, maxlen = 0;
        for(int right = 0; right<s.length(); right++){
            if(ans.count(s[right]) && ans[s[right]] >= left){
                left = ans[s[right]] +1;

            }
            ans[s[right]] = right;
            maxlen = max(maxlen, right-left+1);
        }
        return maxlen;

    }
};
