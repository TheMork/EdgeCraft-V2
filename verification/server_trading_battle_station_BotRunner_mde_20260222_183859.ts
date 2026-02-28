                        type: pos.side === 'long' ? 'LONG' : 'SHORT',
                        entryPrice: pos.entryPrice || 0,
                        stopLoss: 0,
                        originalSL: 0,
                        quantity: Math.abs(pos.contracts),
                        status: 'OPEN',
                        entryTime: Date.now(),
                        highestPrice: pos.entryPrice || 0,
                        lowestPrice: pos.entryPrice || 0
                    };
                    this.state.activeTrades.push(existingTrade);
                } else {
                    // Update existing trade with real exchange data
                    existingTrade.quantity = Math.abs(pos.contracts);
                    if (pos.entryPrice && Math.abs(existingTrade.entryPrice - pos.entryPrice) > 0.01) {
                        existingTrade.entryPrice = pos.entryPrice;
                    }
                }
            }

            // Also check for orphaned trades in state that have no position on exchange
            for (let i = this.state.activeTrades.length - 1; i >= 0; i--) {
                const trade = this.state.activeTrades[i];
                const hasPos = openPositions.some(p => p.symbol === trade.symbol || p.symbol.startsWith(trade.symbol.split('/')[0]));
                if (!hasPos) {
                    console.log(`[Sync] Removing orphaned trade from state: ${trade.symbol}`);
                    this.state.activeTrades.splice(i, 1);
                }
            }
        } catch (e) {
            console.error(`[Sync] Error during global sync:`, e);
        }
    }

    /**
     * Rebalancing logic for mde_mad_v2 (matches EdgeCraft Python behavior).
     * On each tick: compute target weight → compare with current position → adjust.
     * No explicit stop loss — the strategy self-corrects via weight changes.
     */
    async rebalanceSymbol(symbol: string) {
        // 1. Fetch candles
        const candlesLow = await this.client.fetchCandles(symbol, CONFIG.lowTF, 300);
        if (candlesLow.length < 201) {
            if (!this.state.monitoringQueue) this.state.monitoringQueue = {};
            this.state.monitoringQueue[symbol] = { status: 'ERROR', details: 'Insufficient data', lastChecked: Date.now() };
            return;
        }

        // 2. Get strategy signal with target weight
        const analysis = this.engine.analyze([], candlesLow);
        const targetWeight = analysis.signal?.targetWeight ?? 0;

        // Update monitoring queue
        if (!this.state.monitoringQueue) this.state.monitoringQueue = {};

        // Accurate status based on real positions
        let status: string = analysis.status;
        const hasTrade = this.state.activeTrades.some(t => t.symbol === symbol);
        if (hasTrade) {
            status = 'IN_TRADE';
        } else if (status === 'ENTRY' || (status === 'REBALANCE' && Math.abs(targetWeight) > 0.005)) {
            status = 'PENDING';
        }

        this.state.monitoringQueue[symbol] = {
            status,
            details: analysis.details || `w=${targetWeight.toFixed(3)}`,
            lastChecked: Date.now()
        };

        // 3. Get current balance and position
        const balanceInfo = await this.client.getBalance();
        const equity = balanceInfo?.total?.USDT || 1000;
        const currentPrice = await this.client.getPrice(symbol);

        // Target notional = equity × targetWeight
        const targetQty = (equity * targetWeight) / currentPrice;

        // 4. Get current position from state (synced by syncAllPositions)
        const existingTrade = this.state.activeTrades.find(t => t.symbol === symbol);
        let currentQty = 0;
        if (existingTrade) {
            currentQty = existingTrade.type === 'LONG' ? existingTrade.quantity : -existingTrade.quantity;

            // Calculate live PnL for UI
            if (existingTrade.type === 'LONG') {
                existingTrade.pnl = (currentPrice - existingTrade.entryPrice) * existingTrade.quantity;
                existingTrade.pnlPercent = ((currentPrice - existingTrade.entryPrice) / existingTrade.entryPrice) * 100;
            } else {
                existingTrade.pnl = (existingTrade.entryPrice - currentPrice) * existingTrade.quantity;
                existingTrade.pnlPercent = ((existingTrade.entryPrice - currentPrice) / existingTrade.entryPrice) * 100;
            }

            // Update SL/TP placeholders if needed or keep at 0 for rebalancing
            existingTrade.highestPrice = Math.max(existingTrade.highestPrice || 0, currentPrice);
            existingTrade.lowestPrice = Math.min(existingTrade.lowestPrice || currentPrice, currentPrice);
        }
        const currentNotional = currentQty * currentPrice;
        const targetNotional = equity * targetWeight;

        // 5. Calculate diff
        const diffNotional = targetNotional - currentNotional;
        const diffQty = Math.abs(diffNotional / currentPrice);
        const minRebalanceThreshold = equity * 0.005; // 0.5% of equity minimum change

        if (Math.abs(diffNotional) < minRebalanceThreshold) {
            return;
        }

        console.log(`[Rebalance] ${symbol}: w=${targetWeight.toFixed(3)} | target=$${targetNotional.toFixed(2)} | current=$${currentNotional.toFixed(2)} | diff=$${diffNotional.toFixed(2)}`);

        // 6. Execute rebalance
        try {
            if (Math.abs(targetWeight) < 0.005) {
                // Close entire position
                if (currentQty !== 0) {
                    const side: 'buy' | 'sell' = currentQty > 0 ? 'sell' : 'buy';
                    await this.client.createMarketOrder(symbol, side, Math.abs(currentQty));
                    if (existingTrade) {
                        existingTrade.status = 'CLOSED';
                        existingTrade.exitTime = Date.now();
                        existingTrade.exitPrice = currentPrice;

                        // Finalize PnL
                        if (existingTrade.type === 'LONG') {
                            existingTrade.pnl = (currentPrice - existingTrade.entryPrice) * existingTrade.quantity;
                        } else {
                            existingTrade.pnl = (existingTrade.entryPrice - currentPrice) * existingTrade.quantity;
                        }

                        this.state.tradeHistory.unshift(existingTrade);
                        this.state.activeTrades = this.state.activeTrades.filter(t => t.id !== existingTrade.id);
                    }
                    console.log(`[Rebalance] ${symbol}: Closed position (w→0)`);
                }
            } else if (currentQty === 0) {
                // Open new position
                const side: 'buy' | 'sell' = targetWeight > 0 ? 'buy' : 'sell';
                const qty = parseFloat(diffQty.toFixed(3));
                if (qty > 0) {
                    await this.client.createMarketOrder(symbol, side, qty);
                    const trade: Trade = {
                        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                        symbol,
                        type: targetWeight > 0 ? 'LONG' : 'SHORT',
                        entryPrice: currentPrice,
                        stopLoss: 0,
                        originalSL: 0,
                        quantity: qty,
                        status: 'OPEN',
                        entryTime: Date.now(),
                        highestPrice: currentPrice,
                        lowestPrice: currentPrice,
                    };
                    this.state.activeTrades.push(trade);
                    console.log(`[Rebalance] ${symbol}: Opened ${trade.type} qty=${qty} w=${targetWeight.toFixed(3)}`);
                }
            } else if (existingTrade) {
                // Adjust existing position
                if ((targetWeight > 0 && existingTrade.type === 'SHORT') || (targetWeight < 0 && existingTrade.type === 'LONG')) {
                    // Direction change: close old, open new
                    const qtyToClose = existingTrade.quantity;
                    const sideToClose: 'buy' | 'sell' = existingTrade.type === 'LONG' ? 'sell' : 'buy';
                    await this.client.createMarketOrder(symbol, sideToClose, qtyToClose);

                    // Close old in state
                    existingTrade.status = 'CLOSED';
                    existingTrade.exitTime = Date.now();
                    existingTrade.exitPrice = currentPrice;
                    this.state.tradeHistory.unshift(existingTrade);
                    this.state.activeTrades = this.state.activeTrades.filter(t => t.id !== existingTrade.id);

                    // Open new
                    const sideNew: 'buy' | 'sell' = targetWeight > 0 ? 'buy' : 'sell';
                    const qtyNew = parseFloat(Math.abs(targetQty).toFixed(3));
                    if (qtyNew > 0) {
                        await this.client.createMarketOrder(symbol, sideNew, qtyNew);
                        const newTrade: Trade = {
                            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                            symbol,
                            type: targetWeight > 0 ? 'LONG' : 'SHORT',
                            entryPrice: currentPrice,
                            stopLoss: 0,
                            originalSL: 0,
                            quantity: qtyNew,
                            status: 'OPEN',
                            entryTime: Date.now(),
                            highestPrice: currentPrice,
                            lowestPrice: currentPrice,
                        };
                        this.state.activeTrades.push(newTrade);
