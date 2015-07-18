import org.jblas.FloatMatrix;
import org.jblas.util.Random;


/**
 * Created by Tom on 7/14/2015.
 */
public class SingleLayer {
    public static void main(final String[] args) {
        final FloatMatrix testData = new FloatMatrix(new float[][]{
                new float[]{0, 0, 1},
                new float[]{0, 1, 1},
                new float[]{1, 0, 1},
                new float[]{1, 1, 1},
        });
        final FloatMatrix testResults = new FloatMatrix(new float[][] {
                new float[]{0},
                new float[]{0},
                new float[]{1},
                new float[]{1},
        });

        // Set our random seed
        Random.seed(1);

        final FloatMatrix synapse0 = FloatMatrix.rand(3, 1).mul(2).sub(1);
        FloatMatrix layer1 = null;
        for (int i = 0; i < 10000; i++) {
            final FloatMatrix layer0 = testData.dup();
            layer1 = nonLinear(layer0.mmul(synapse0));
            final FloatMatrix layer1Error = testResults.sub(layer1);
            final FloatMatrix layer1Changes = layer1Error.mul(nonLinearDeriv(layer1));
            synapse0.addi(layer0.transpose().mmul(layer1Changes));
        }

        System.out.println("Layer 1: " + layer1);
        System.out.println("Synapse 0: " + synapse0);
    }

    public static FloatMatrix nonLinear(final FloatMatrix data) {
        final FloatMatrix result = new FloatMatrix(data.rows, data.columns);

        for (int row = 0; row < result.rows; row++) {
            for (int col = 0; col < result.columns; col++) {
                result.put(row, col, 1f / (1f + (float)Math.exp(-data.get(row, col))));
            }
        }

        return result;
    }

    public static FloatMatrix nonLinearDeriv(final FloatMatrix data) {
        return new FloatMatrix(data.rows, data.columns).fill(1).sub(data).mul(data);
    }
}
