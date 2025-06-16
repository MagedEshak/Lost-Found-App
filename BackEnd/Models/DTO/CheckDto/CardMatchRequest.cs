using Lost_and_Found.Enums;
using Lost_and_Found.Models.DTO.CheckDto;
using Newtonsoft.Json;
using MatchType = Lost_and_Found.Enums.MatchType;

namespace Lost_and_Found.Models.DTO
{

    public class CardMatchRequest
    {
        [JsonProperty("match_type")]

        public MatchType MatchType { get; set; }
      
        [JsonProperty("lost")]
        public CardLostMatchDto Lost { get; set; }
    }

   
}
