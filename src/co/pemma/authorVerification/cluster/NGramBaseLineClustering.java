package co.pemma.authorVerification.cluster;

import java.util.List;
import java.util.Map;

import co.pemma.authorVerification.NGramVectors;
import co.pemma.authorVerification.NGramVectors.TokenType;

public class NGramBaseLineClustering extends BaseLineClustering
{
	public NGramBaseLineClustering(TokenType token, Map<String, String> posts, List<String> clusterPosts)
	{
		super(posts, clusterPosts, true);

		// generate the vectors for the problem set and imposters
		this.vectorMap = NGramVectors.run(posts, token);
	}	
}
