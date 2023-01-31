/*
 * MuratCanvas2dVerticalView.h
 *
 *  Created on: 16 janv. 2018
 *      Author: j0483271
 */

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DVERTICALVIEW_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DVERTICALVIEW_H_

#include "MuratCanvas2dView.h"

#include <QWidget>


class MuratCanvas2dVerticalView: public MuratCanvas2dView {
	Q_OBJECT
public:
	MuratCanvas2dVerticalView(QWidget* parent=0);
	virtual ~MuratCanvas2dVerticalView();

	void initScene();
	virtual void scale(qreal, qreal) override;

	qreal getYRatio();

	void setImage(QImage img, Matrix2DInterface* mat=0);



public slots:
	/*
	 * Slot that scale * 2.0 in y axis
	 */
	void verticalZoomIn();

	/*
	 * Slot that scale / 2.0 in y axis
	 */
	void verticalZoomOut();

	/*
     * Slot to scale image to given ratio in y axis
	 */
	void setYRatio(qreal yRatio);

	void setXRatio(qreal xRatio);

	void simulate_gentle_y_move(qreal yPos);

	/*
	 * Public method
	 *
	 * Do zoom taking into account previous wheel event, centering on mouse position
	 */
	virtual void gentle_zoom(double factor);

	/*
	 * Public method
	 *
	 * Do zoom taking into account previous wheel event, centering on position
	 */
	virtual void simulate_gentle_zoom(double factor, qreal y_target_scene_pos);

protected:
	/*
	 * Protected Method that manage zoom through wheel
	 */
	virtual void mouseMoveEventWheelManager(QMouseEvent *event);

	void initLogMode();
	void initImageMode();


private:
	/*
     * Private method
     *
     * Getter of canvas should be use to access canvas
     */
	MuratCanvas2dScene* getCanvas();
	qreal xRatio;
	qreal yRatio;
	qreal y_target_viewport_pos, y_target_scene_pos;
};

#endif /* MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DVERTICALVIEW_H_ */
