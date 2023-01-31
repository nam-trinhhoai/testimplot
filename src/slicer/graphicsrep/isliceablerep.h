#ifndef ISliceableRep_H
#define ISliceableRep_H

#include <QString>

/*
 * A graphic rep who need to be aware about the slice position
 */
class ISliceableRep {
public:
	ISliceableRep() {
	}
	virtual ~ISliceableRep() {
	}
	virtual void setSliceIJPosition(int val) =0;
};
#endif
