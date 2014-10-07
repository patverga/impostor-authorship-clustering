package co.pemma.authorVerification.cluster;

import java.util.List;
import java.util.Map;
import co.pemma.authorVerification.WritePrintsVectors;

public class WritePrintsBaseLineClustering extends BaseLineClustering
{
	public WritePrintsBaseLineClustering(Map<String, String> posts, List<String> clusterPostsList)
	{
		super(posts, clusterPostsList, false);
		this.vectorMap = WritePrintsVectors.jstyloVectors(posts, clusterPostsList);	
		this.clusterPosts = vectorMap.keySet().toArray(new String[vectorMap.size()]);
	}		
}
