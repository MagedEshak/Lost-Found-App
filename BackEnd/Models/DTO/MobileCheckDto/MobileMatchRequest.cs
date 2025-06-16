using Lost_and_Found.Enums;
using Lost_and_Found.Models.DTO.CheckDto;
using Newtonsoft.Json;
using MatchType = Lost_and_Found.Enums.MatchType;

namespace Lost_and_Found.Models.DTO.MobileCheckDto
{

    public class MobileMatchRequest
    {
        [JsonProperty("match_type")]
        public MatchType MatchType { get; set; }
            [JsonProperty("lost")]
        public MobileLostMatchDto Lost { get; set; }
    }


}
