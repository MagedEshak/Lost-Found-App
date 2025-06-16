using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Lost_and_Found.Models.Entites
{
    public class LostCard
    {
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        [Key]

        public int Id { get; set; }
        public string CardID { get; set; }
        public byte[]? CardPhoto { get; set; }
        public string ImageName { get; set; }

        public string Government { get; set; }
        public string Center { get; set; }
        public string Street { get; set; }
        public string ForiegnKey_UserEmail { get; set; }
        public User User { get; set; }                  // One To Many relation
    }
}
