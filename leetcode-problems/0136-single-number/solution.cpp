class Solution {
public:
    int singleNumber(vector<int>& nums) {
        // unordered_map<int, int> freq;
        // for(int num: nums){
        //     freq[num]++;
        // }
        // for(auto pair: freq){
        //     if(pair.second == 1){
        //         return pair.first;
        //     }
        // }
        // return -1;
        int result = 0;
        for(int num: nums){
            result ^= num;
        }
        return result;
        

    }
};
