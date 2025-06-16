using Newtonsoft.Json;

namespace Lost_and_Found.Models.DTO.CheckDto
{
    public class CardLostMatchDto
    {
        [JsonProperty("name")]
        public string name { get; set; }

        [JsonProperty("national_id")]
        public string national_id { get; set; }

        [JsonProperty("governorate")]
        public string governorate { get; set; }

        [JsonProperty("city")]
        public string city { get; set; }

        [JsonProperty("street")]
        public string street { get; set; }

        [JsonProperty("contact")]
        public string contact { get; set; }

        [JsonProperty("image_name")]
        public string image_name {get;set;}
    }
}
