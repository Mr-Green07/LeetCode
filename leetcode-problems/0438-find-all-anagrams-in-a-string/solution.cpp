class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int k = p.length();
        int n = s.length();
        if(n<k){
            return {};
        }
        vector<int> result;
        vector<int> s_freq(26, 0);
        vector<int> p_freq(26, 0);

        for(int i =0; i<k; i++){
            s_freq[s[i] - 'a']++;
            p_freq[p[i] - 'a']++;
        }
        if(p_freq == s_freq){
            result.push_back(0);
        }
        for(int i = k; i<n;i++){
            s_freq[s[i] - 'a']++;
        
            s_freq[s[i-k] - 'a']--;
            if(p_freq == s_freq){
                result.push_back(i - k + 1);
            }
        }
        return result;
    }
};
