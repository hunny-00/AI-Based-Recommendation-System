import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class RecommenderSystem {
    public static void main(String[] args) {
        try {
            // Step 1: Load the data
            File file = new File("src/main/resources/dataset.csv");
            DataModel model = new FileDataModel(file);

            // Step 2: Calculate similarity between users
            UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

            // Step 3: Define Neighborhood
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

            // Step 4: Build the Recommender
            Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

            // Step 5: Generate Recommendations for All Users
            for (LongPrimitiveIterator users = model.getUserIDs(); users.hasNext(); ) {
                long userId = users.nextLong();
                List<RecommendedItem> recommendations = recommender.recommend(userId, 3); // Top 3

                System.out.println("User " + userId + " Recommendations:");
                for (RecommendedItem item : recommendations) {
                    System.out.println("  Item: " + item.getItemID() + ", Estimated Preference: " + item.getValue());
                }
                System.out.println();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
