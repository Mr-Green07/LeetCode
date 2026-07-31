class Solution {
public:
    int minimumPushes(string word) {
        vector<int> count(26, 0);
        for(auto it : word){
            count[it - 'a']++;
        }
        sort(count.rbegin(), count.rend());
        int pushes = 0;
        for(int i = 0; i< 26; i++){
            pushes += (i / 8 + 1) * count[i];
        }
        return pushes;
    }
};
