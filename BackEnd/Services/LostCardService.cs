using Lost_and_Found.Models.DTO;
using Lost_and_Found.Models.Entites;
using Lost_and_Found.Models;
using AutoMapper;
using Lost_and_Found.Interfaces;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Linq;
using Microsoft.EntityFrameworkCore;

namespace Lost_and_Found.Services
{
    public class LostCardService:ILostCardService
    {
        private DataConnection con;
        private IMapper mp;
        public LostCardService(DataConnection con, IMapper mp)
        {
            this.con = con;
            this.mp = mp;
        }

        public List<LostCard> GetLostCards()
        {
            return con.LostCards.ToList();
        }

        public List<LostCard> GetLostCardsOfEmail(string email)
        {
            return con.LostCards
           .Include(o => o.User)
           .Where(o => o.ForiegnKey_UserEmail == email)
           .ToList();
        }
        public async Task<LostCard?> AddLostCard(LostCardsDTO lostCardDTO)
        {
            //if (con.LostCards.Any(o => o.CardID == lostCardDTO.CardID) ||
            //    !con.Users.Any(o => o.Email == lostCardDTO.ForiegnKey_UserEmail))
            //{
            //    return null;
            //}

            byte[]? photoBytes = null;
            if (lostCardDTO.CardPhoto != null)
            {
                var stream = new MemoryStream();
                await lostCardDTO.CardPhoto.CopyToAsync(stream);
                photoBytes = stream.ToArray();
            }

            string prefix = "lost";
            string uniquePart = Guid.NewGuid().ToString("N");
            string imageName = prefix + uniquePart + ".jpg";

            LostCard lostCard = new()
            {
                CardID = lostCardDTO.CardID,
                CardPhoto = photoBytes,
                Street = lostCardDTO.Street,
                Center = lostCardDTO.Center,
                Government = lostCardDTO.Government,
                ImageName = imageName,
                ForiegnKey_UserEmail = lostCardDTO.ForiegnKey_UserEmail
            };
            //
            try
            {
                con.LostCards.Add(lostCard);
                await con.SaveChangesAsync();
                
            }
           catch (Exception ex)
            {
                Console.WriteLine("Error: " + ex.Message);
            }

            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromMinutes(5);

            var form = new MultipartFormDataContent();

            if (photoBytes != null && photoBytes.Length > 0)
            {
                var imageContent = new ByteArrayContent(photoBytes);
                imageContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
                form.Add(imageContent, "image", imageName);
            }

            form.Add(new StringContent(lostCard.ForiegnKey_UserEmail ?? "Unknown"), "name");
            form.Add(new StringContent(lostCard.CardID ?? ""), "national_id");
            form.Add(new StringContent(lostCard.Government ?? ""), "governorate");
            form.Add(new StringContent(lostCard.Center ?? ""), "city");
            form.Add(new StringContent(lostCard.Street ?? ""), "street");
            form.Add(new StringContent(lostCard.ForiegnKey_UserEmail ?? ""), "contact");
            form.Add(new StringContent(lostCard.ImageName ?? ""), "image_name");


            try
            {
                var response = await httpClient.PostAsync("http://localhost:8000/add_lost_card", form);
                var responseContent = await response.Content.ReadAsStringAsync();

                Console.WriteLine($"Status: {response.StatusCode}");
                Console.WriteLine($"Response: {responseContent}");

                if (!response.IsSuccessStatusCode)
                {
                    // طباعة تفاصيل أكثر للخطأ
                    Console.WriteLine($"Request failed with status: {response.StatusCode}");
                    Console.WriteLine($"Response content: {responseContent}");

                    // محاولة parse الخطأ من FastAPI
                    try
                    {
                        dynamic errorDetails = Newtonsoft.Json.JsonConvert.DeserializeObject(responseContent);
                        Console.WriteLine($"Error details: {errorDetails}");
                    }
                    catch
                    {
                        Console.WriteLine("Could not parse error response");
                    }

                    throw new Exception($"Failed to send data to AI service. Status: {response.StatusCode}, Response: {responseContent}");
                }

                return lostCard;
            }
            catch (HttpRequestException httpEx)
            {
                Console.WriteLine($"HTTP Request Error: {httpEx.Message}");
                throw new Exception($"Network error when calling AI service: {httpEx.Message}");
            }
            catch (TaskCanceledException tcEx)
            {
                Console.WriteLine($"Request Timeout: {tcEx.Message}");
                throw new Exception("Request to AI service timed out");
            }
        }


        public LostCard UpdateLostCard(LostCardsDTO card)
        {
            if (con.LostCards.FirstOrDefault(o => o.CardID == card.CardID) == null
                || con.Users.FirstOrDefault(o => o.Email == card.ForiegnKey_UserEmail) == null)
                return null;

            LostCard card1 = con.LostCards.FirstOrDefault(o => o.CardID == card.CardID);

            using var stream = new MemoryStream();
            card.CardPhoto?.CopyTo(stream);


            card1.CardID = card.CardID;
            card1.CardPhoto = stream.ToArray();
            card1.Street = card.Street;
            card1.Center = card.Center;
            card1.Government = card.Government;

            con.LostCards.Update(card1);
            con.SaveChanges();

            return card1;
        }

        public string DeleteLostCard(string email, string cardid)
        {
            if (con.LostCards.FirstOrDefault(o => o.ForiegnKey_UserEmail == email) == null
                || con.LostCards.FirstOrDefault(o => o.CardID == cardid) == null)
                return null;

            con.LostCards.Remove(con.LostCards.FirstOrDefault(o => o.CardID == cardid));
            con.SaveChanges();
            return $"Card Number {cardid} Deleted";
        }

      
    }
}
