class Solution {
public:
    vector<int> minDistinctFreqPair(vector<int>& nums) {
        vector<int> freq(101, 0);
        for(int num: nums){
            freq[num]++;
        }
        vector<int> values;
        for(int i = 1; i <= 100; i++){
            if(freq[i]> 0){
                values.push_back(i);
            }
        }
        if(values.size()< 2){
            return{ -1, -1};
        }
        for(int i = 0; i<values.size(); i++){
            for(int j = i + 1; j< values.size();  j++){
                if(freq[values[i]] != freq[values[j]]){
                    return{values[i], values[j]};
                }
            }
        }
        return {-1, -1};
            
    }
};
