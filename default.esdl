using extension pgvector;

module default {
     type Recipe{
        required property name -> str; 
        property RecipeId -> int32;
        property TotalTime -> int32;
        property Description -> str;
        property Keywords -> str;
        property RecipeCategory -> str;
        property ingredients -> str;
        property Calories -> float32;
        property FatContent -> float32;
        property SaturatedFatContent -> float32;
        property CholestrerolContent -> float32;
        property SodiumContent -> float32;
        property CarboyhdrateContent -> float32;
        property FiberContent -> float32;
        property SugarContent -> float32;
        property ProteinContent -> float32;
        property Popularity -> str;
        property ingredient_num -> int32;
        property Calories_PerServing -> float32;
        property FatContent_PerServing -> float32;
        property SaturatedFatContent_PerServing -> float32;
        property CholesterolContent_PerServing -> float32;
        property SodiumContent_PerServing -> float32;
        property CarbohydrateContent_PerServing -> float32;   
        property FiberContent_PerServing -> float32;
        property SugarContent_PerServing -> float32;
        property ProteinContent_PerServing -> float32;            
        property RecipeInstructions -> str;
        multi link review -> Review {
            property review_id -> int32;
        };
    }

        type Review{
        required property RecipeId -> int32; 
        property Rating -> float32;
        property Review -> str;
        
    }


scalar type v3 extending ext::pgvector::vector<3>;
scalar type v100 extending ext::pgvector::vector<100>;

        type RecipeEmbedding{
        required property RecipeId -> int32; 
        property Name -> str;
        property cleaned_ingre -> str;
        property combined -> str;
        embedding : v3;
        embeddings : v100;
    }

        type Ingredient {
        required property Ingredient -> str; 
        property Season -> str;
       
    }

        type Spice {
        required property Spice -> str; 
        property Description -> str;
       
    }


}
