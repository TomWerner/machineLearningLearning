import org.jblas.FloatMatrix;
import org.jblas.util.Random;


/**
 * Created by Tom on 7/14/2015.
 */
public class TwoLayers {
    public static void main(final String[] args) {
        final FloatMatrix testData = new FloatMatrix(new float[][]{
                new float[]{0, 0, 1},
                new float[]{0, 1, 1},
                new float[]{1, 0, 1},
                new float[]{1, 1, 1},
        });
        final FloatMatrix testResults = new FloatMatrix(new float[][] {
                new float[]{0},
                new float[]{1},
                new float[]{1},
                new float[]{0},
        });

        // Set our random seed
        Random.seed(1);

        final FloatMatrix synapse0 = FloatMatrix.rand(3, 4).mul(2).sub(1);
        final FloatMatrix synapse1 = FloatMatrix.rand(4, 1).mul(2).sub(1);

        FloatMatrix layer0 = testData.dup();
        FloatMatrix layer1 = null;
        FloatMatrix layer2 = null;
        FloatMatrix layer1Error;
        FloatMatrix layer2Error;
        FloatMatrix layer1Changes;
        FloatMatrix layer2Changes;

        for (int i = 0; i < 60000; i++) {
            // Set up our layers
            layer1 = nonLinear(layer0.mmul(synapse0));
            layer2 = nonLinear(layer1.mmul(synapse1));

            // Check how close we were
            layer2Error = testResults.sub(layer2);

            if ((i % 10000) == 0) {
                System.out.println("Error: " + absi(layer2Error).mean());
            }

            // Figure out how sure we were
            layer2Changes = layer2Error.mul(nonLinearDeriv(layer2));

            // How much did this layer contribute to the error by the weights
            layer1Error = layer2Changes.mmul(synapse1.transpose());

            // In what direction is the target layer1?
            layer1Changes = layer1Error.mul(nonLinearDeriv(layer1));

            synapse1.addi(layer1.transpose().mmul(layer2Changes));
            synapse0.addi(layer0.transpose().mmul(layer1Changes));
        }

        System.out.println("Layer 2: " + layer2);
        System.out.println("Synapse 1: " + synapse1);
        System.out.println("Layer 1: " + layer1);
        System.out.println("Synapse 0: " + synapse0);
        System.out.println("Layer 0: " + layer0);
    }

    public static FloatMatrix absi(final FloatMatrix data) {
        final FloatMatrix result = data.dup();
        for (int row = 0; row < result.rows; row++) {
            for (int col = 0; col < result.columns; col++) {
                result.put(row, col, Math.abs(result.get(row, col)));
            }
        }
        return result;
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
