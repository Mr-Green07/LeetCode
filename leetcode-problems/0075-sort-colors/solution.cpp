class Solution {
public:
    void sortColors(vector<int>& nums) {
        vector<int> freq(3, 0);
        for (int num: nums){
            freq[num]++;
        }
        int i = 0;
        for(int color = 0; color <= 2; color++){
            while(freq[color] > 0){
                nums[i] = color;
                i++;
                freq[color]--;
            }
        }
        
    }
};
