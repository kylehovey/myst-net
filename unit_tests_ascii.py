from unit_tests import printESC, printColor, testLoadImage, testProcessImage, testLoadData, testLoadNet, testRunNet, testTestingAccuracy, testValidationAccuracy

# <=< Testing Utility Functions >=>
class Test():
    def __init__(self):
        self.tests = []
    def it(self, description, name, callback):
        def run():
            success = callback()

            if success:
                printColor("It {}: [PASS]".format(description), "38;32m")
            else:
                printColor("It {}: [FAIL]".format(description), "38;31m")

        self.tests.append((run, name))
    def runTests(self):
        nTests = len(self.tests)
        print("{} tests to run".format(nTests))
        print("<==================>")
        for i, (test, name) in enumerate(self.tests):
            print("Running test {} of {}: {}".format(i + 1, nTests, name))
            test()
            print("-------------------")

if __name__ == "__main__":
    testing = Test()

    testing.it("loads images from the filesystem", "Image Loading", testLoadImage)
    testing.it("processes a loaded image for the network", "Image Processing", testProcessImage)
    testing.it("loads and formats all training and validation data", "Data Loading", testLoadData)
    testing.it("loads the network itself", "Network Loading", testLoadNet)
    testing.it("runs the network on input data", "Network Running", testRunNet)
    testing.it("is at least 95% accurate on testing data", "Testing Accuracy", testTestingAccuracy)
    testing.it("is at least 95% accurate on validation data", "validation Accuracy", testValidationAccuracy)

    testing.runTests()
