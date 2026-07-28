class Solution {
public:
    string smallestPalindrome(string s) {
        int n = s.length();
        int freq[26] = {0};
        string sortedhalf;
        for(int i = 0; i<n/2; i++){
            freq[s[i] - 'a']++;
        }
        for(int i = 0; i<26; i++){
            sortedhalf += string(freq[i] , 'a' + i);
        }
        
        string right = sortedhalf;
        reverse(right.begin(), right.end());

        string mid = (n % 2)? string(1, s[n/2]):"";

        return sortedhalf + mid + right;
    }
};
