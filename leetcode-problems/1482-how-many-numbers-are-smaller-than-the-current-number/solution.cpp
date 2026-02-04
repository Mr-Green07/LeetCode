class Solution {
public:
    vector<int> smallerNumbersThanCurrent(vector<int>& nums) {
    //     vector<int> result;
    //     for(int i = 0; i< nums.size(); i++){
    //         int count = 0;
    //         for (int j = 0; j<nums.size(); j++){
    //             if(nums[i]> nums[j]){
    //                 count++;
    //             }
    //         }
    //         result.push_back(count);
    //     }
    //     return result;
    // }

        vector<int> freq(101, 0);
        vector<int> result;

        for(int num: nums){
            freq[num]++;
        }

        for (int i = 1; i<101; i++){
            freq[i] += freq[i -1];
        }
        for (int num: nums){
            if (num == 0){
                result.push_back(0);
            }else{
                result.push_back(freq[num-1]);
            }
        }
    return result;
    }
};
