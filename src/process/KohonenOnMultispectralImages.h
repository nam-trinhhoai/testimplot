#ifndef KohonenOnMultispectralImages_H_
#define KohonenOnMultispectralImages_H_

#include <cstddef>
#include <vector>

#include <QString>

class CUDAImagePaletteHolder;
class RgbLayerFromDataset;
class FixedLayerFromDataset;

class KohonenImage2D {
public:
	KohonenImage2D();
	virtual ~KohonenImage2D();

	virtual bool isExampleValid(long i, long j) const = 0;
	virtual bool getExample(float* tab, long i, long j) const = 0;

	virtual std::size_t dimI() const = 0;
	virtual std::size_t dimJ() const = 0;
	virtual std::size_t exampleSize() const = 0;
};

class KohonenCudaPlanarImage2D : public KohonenImage2D {
public:
	KohonenCudaPlanarImage2D(const std::vector<const float*>& stack, std::size_t dimI, std::size_t dimJ, short nonValue);
	virtual ~KohonenCudaPlanarImage2D();

	virtual bool isExampleValid(long i, long j) const override;
	virtual bool getExample(float* tab, long i, long j) const override;

	virtual std::size_t dimI() const override;
	virtual std::size_t dimJ() const override;
	virtual std::size_t exampleSize() const override;

private:
	std::vector<const float*> m_stack;
	std::size_t m_dimI;
	std::size_t m_dimJ;
	short m_nonValue;
};

class KohonenOnMultispectralImages {
public:
	// tmapOrPca : true for tmap false for pca
	KohonenOnMultispectralImages(KohonenImage2D* image, bool tmapOrPca, bool takeImageOwnership);
	~KohonenOnMultispectralImages();

	bool setOutputHorizonProperties(RgbLayerFromDataset* saveProperties, QString tmapLabel);
	bool setOutputHorizonProperties(FixedLayerFromDataset* saveProperties, QString tmapLabel);

	bool compute(int kohonenMapSize, int trainExamplesStep);

private:
	KohonenImage2D* m_image;
	bool m_isOwnerOfImage;

	RgbLayerFromDataset* m_computationResultRgb = nullptr;
	FixedLayerFromDataset* m_computationResultGray = nullptr;
	QString m_tmapLabel;
	bool m_tmapOrPca;
};

#endif
