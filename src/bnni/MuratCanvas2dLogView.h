/*
 * MuratCanvas2dLogView.h
 *
 *  Created on: 26 janv. 2018
 *      Author: j0483271
 */

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DLOGVIEW_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DLOGVIEW_H_

#include "MuratCanvas2dVerticalView.h"

#include <QWidget>

class MuratCanvas2dLogView: public MuratCanvas2dVerticalView {
	Q_OBJECT
public:
	MuratCanvas2dLogView(QWidget* parent=0);
	virtual ~MuratCanvas2dLogView();

	void initScene();

    void setImage(QImage img, Matrix2DInterface* mat = nullptr);
    std::vector<double> getLines();
    void setLines(const std::vector<double>&);
    void scale(qreal sx, qreal sy);

private:
	/*
     * Private method
     *
     * Getter of canvas should be use to access canvas
     */
    MuratCanvas2dScene* getCanvas();

signals:
    void linesChanged(const std::vector<double> lines);
};


#endif /* MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DLOGVIEW_H_ */
